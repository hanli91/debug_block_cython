from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set as cset
from libcpp.unordered_map cimport unordered_map as cmap
from libcpp.queue cimport priority_queue as heap
from libcpp.string cimport string
from libcpp cimport bool
from libc.stdio cimport printf, fprintf, fopen, fclose, FILE, sprintf
from libc.stdint cimport uint32_t as uint, uint64_t
from cython.parallel import prange, parallel
import time

include "new_topk_sim_join.pyx"
include "original_topk_sim_join.pyx"


# cdef extern from "TopPair.h" nogil:
#     cdef cppclass TopPair nogil:
#         TopPair()
#         TopPair(double, uint, uint)
#         double sim
#         uint l_rec
#         uint r_rec
#         bool operator>(const TopPair& other)
#         bool operator<(const TopPair& other)
#
#
# cdef extern from "PrefixEvent.h" nogil:
#     cdef cppclass PrefixEvent:
#         PrefixEvent()
#         PrefixEvent(double, int, int, int)
#         double threshold
#         int table_indicator
#         int rec_idx
#         int tok_idx
#         bool operator>(const PrefixEvent& other)
#         bool operator<(const PrefixEvent& other)


PREFIX_MATCH_MAX_SIZE = 4
PREFIX_MULTIPLY_FACTOR = 5
OFFSET_OF_FIELD_NUM = 10
MINIMAL_NUM_FIELDS = 0
FIELD_REMOVE_RATIO = 0.1

def debugblocker_cython(lrecord_token_list, rrecord_token_list,
                        lrecord_index_list, rrecord_index_list,
                        ltable_field_token_sum, rtable_field_token_sum, py_cand_set,
                        py_num_fields, py_output_size, py_output_path, py_use_plain):
    cdef string output_path = py_output_path
    cdef bool use_plain = py_use_plain

    ### Convert py objs to c++ objs
    cdef vector[vector[int]] ltoken_vector, rtoken_vector
    convert_table_to_vector(lrecord_token_list, ltoken_vector)
    convert_table_to_vector(rrecord_token_list, rtoken_vector)

    cdef vector[vector[int]] lindex_vector, rindex_vector
    convert_table_to_vector(lrecord_index_list, lindex_vector)
    convert_table_to_vector(rrecord_index_list, rindex_vector)

    cdef vector[int] ltoken_sum, rtoken_sum
    convert_py_list_to_vector(ltable_field_token_sum, ltoken_sum)
    convert_py_list_to_vector(rtable_field_token_sum, rtoken_sum)

    cdef cmap[int, cset[int]] cand_set
    convert_candidate_set_to_c_map(py_cand_set, cand_set)

    cdef vector[int] field_list
    for i in range(py_num_fields):
        field_list.push_back(i)

    cdef uint output_size = py_output_size
    cdef uint prefix_match_max_size = PREFIX_MATCH_MAX_SIZE
    cdef uint prefix_multiply_factor = PREFIX_MULTIPLY_FACTOR
    cdef uint offset_of_field_num = OFFSET_OF_FIELD_NUM
    cdef uint minimal_num_fields = MINIMAL_NUM_FIELDS
    cdef double field_remove_ratio = FIELD_REMOVE_RATIO

    del lrecord_token_list, rrecord_token_list
    del lrecord_index_list, rrecord_index_list
    del py_cand_set

    ### Generate recommendation topk lists
    cdef vector[vector[TopPair]] topk_lists
    cdef cmap[int, cmap[int, ReuseInfo]] reuse_set

    generate_recom_lists(ltoken_vector, rtoken_vector, lindex_vector, rindex_vector,
                         ltoken_sum, rtoken_sum, field_list, topk_lists,
                         cand_set, reuse_set, prefix_match_max_size,
                         prefix_multiply_factor, offset_of_field_num, minimal_num_fields,
                         field_remove_ratio, output_size, output_path, use_plain)


cdef void generate_recom_lists(vector[vector[int]]& ltoken_vector, vector[vector[int]]& rtoken_vector,
                               vector[vector[int]]& lindex_vector, vector[vector[int]]& rindex_vector,
                               const vector[int]& ltoken_sum_vector, const vector[int]& rtoken_sum_vector,
                               vector[int]& field_list, vector[vector[TopPair]]& topk_lists,
                               cmap[int, cset[int]]& cand_set, cmap[int, cmap[int, ReuseInfo]] reuse_set,
                               const uint prefix_match_max_size, const uint prefix_multiply_factor,
                               const uint offset_of_field_num, const uint minimal_num_fields,
                               const double field_remove_ratio, const uint output_size,
                               const string output_path, const bool use_plain):
    if field_list.size() <= minimal_num_fields:
        print 'too few lists:', field_list
        return


    start = time.time()
    generate_recom_list_for_config(ltoken_vector, rtoken_vector,
                                   lindex_vector, rindex_vector,
                                   ltoken_sum_vector, rtoken_sum_vector,
                                   field_list, cand_set, reuse_set,
                                   prefix_match_max_size, prefix_multiply_factor,
                                   offset_of_field_num, output_size, use_plain, 0, output_path)
    end = time.time()
    print 'join time:', end - start

    cdef uint i
    cdef int p
    cdef double max_ratio = 0.0
    cdef uint ltoken_total_sum = 0, rtoken_total_sum = 0
    cdef int removed_field_index = -1
    cdef bool has_long_field = False

    for i in range(field_list.size()):
        ltoken_total_sum += ltoken_sum_vector[field_list[i]]
        rtoken_total_sum += rtoken_sum_vector[field_list[i]]

    cdef double lrec_ave_len = ltoken_total_sum * 1.0 / ltoken_vector.size()
    cdef double rrec_ave_len = rtoken_total_sum * 1.0 / rtoken_vector.size()
    cdef double ratio = 1 - (field_list.size() - 1) * field_remove_ratio / (1.0 + field_remove_ratio) *\
                 double_max(lrec_ave_len, rrec_ave_len) / (lrec_ave_len + rrec_ave_len)

    for i in range(field_list.size()):
        max_ratio = double_max(max_ratio, double_max(ltoken_sum_vector[field_list[i]] * 1.0 / ltoken_total_sum,
                                                     rtoken_sum_vector[field_list[i]] * 1.0 / rtoken_total_sum))
        if ltoken_sum_vector[field_list[i]] > ltoken_total_sum * ratio or\
                rtoken_sum_vector[field_list[i]] > rtoken_total_sum * ratio:
            removed_field_index = i
            has_long_field = True
            break

    if removed_field_index < 0:
        removed_field_index = field_list.size() - 1
    print 'required remove-field ratio:', ratio
    print 'actual max ratio:', max_ratio


    cdef vector[vector[TopPair]] temp_lists
    cdef vector[vector[vector[int]]] ltoken_vector_parallel
    cdef vector[vector[vector[int]]] rtoken_vector_parallel
    cdef vector[vector[int]] field_list_parallel
    cdef vector[int] field_parallel
    cdef vector[int] temp
    if True:
    # if not has_long_field:
        for i in range(field_list.size()):
            if i != removed_field_index:
                temp = vector[int](field_list)
                temp.erase(temp.begin() + i)
                if temp.size() > minimal_num_fields:
                    field_list_parallel.push_back(temp)
                    field_parallel.push_back(field_list[i])
                    ltoken_vector_parallel.push_back(vector[vector[int]]())
                    copy_table_and_remove_field(ltoken_vector, lindex_vector,
                                                ltoken_vector_parallel[ltoken_vector_parallel.size() - 1],
                                                field_list[i])
                    rtoken_vector_parallel.push_back(vector[vector[int]]())
                    copy_table_and_remove_field(rtoken_vector, rindex_vector,
                                                rtoken_vector_parallel[rtoken_vector_parallel.size() - 1],
                                                field_list[i])
        with nogil, parallel(num_threads=field_parallel.size()):
            for p in prange(field_parallel.size()):
                generate_recom_list_for_config(ltoken_vector_parallel[p], rtoken_vector_parallel[p],
                                               lindex_vector, rindex_vector,
                                               ltoken_sum_vector, rtoken_sum_vector,
                                               field_list_parallel[p], cand_set, reuse_set,
                                               prefix_match_max_size, prefix_multiply_factor,
                                               offset_of_field_num, output_size, use_plain, 1, output_path)

    print 'remove', field_list[removed_field_index]
    remove_field(ltoken_vector, lindex_vector, field_list[removed_field_index])
    remove_field(rtoken_vector, rindex_vector, field_list[removed_field_index])
    field_list.erase(field_list.begin() + removed_field_index)

    generate_recom_lists(ltoken_vector, rtoken_vector, lindex_vector, rindex_vector,
                         ltoken_sum_vector, rtoken_sum_vector, field_list, topk_lists, cand_set, reuse_set,
                         prefix_match_max_size, prefix_multiply_factor, offset_of_field_num,
                         minimal_num_fields, field_remove_ratio, output_size, output_path, use_plain)

    return


cdef void generate_recom_list_for_config(const vector[vector[int]]& ltoken_vector, const vector[vector[int]]& rtoken_vector,
                                         const vector[vector[int]]& lindex_vector, const vector[vector[int]]& rindex_vector,
                                         const vector[int]& ltoken_sum_vector, const vector[int]& rtoken_sum_vector,
                                         const vector[int]& field_list, cmap[int, cset[int]]& cand_set,
                                         cmap[int, cmap[int, ReuseInfo]]& reuse_set,
                                         const uint prefix_match_max_size, const uint prefix_multiply_factor,
                                         const uint offset_of_field_num, const uint output_size,
                                         const bool use_plain, const uint type, const string& output_path) nogil:
    cdef uint i, j
    cdef char buf[10]


    cdef string info = string(<char *>'current configuration: [')
    for i in xrange(field_list.size()):
        sprintf(buf, "%d", field_list[i])
        if i != 0:
            info.append(<char *>', ')
        info.append(buf)
    info += <char *>"]  "
    # printf("current configuration: [")
    # for i in xrange(field_list.size()):
    #     if i == 0:
    #         printf("%d", field_list[i])
    #     else:
    #         printf(" %d", field_list[i])
    # printf("]\n")

    cdef int ltoken_total_sum = 0, rtoken_total_sum = 0
    for i in xrange(field_list.size()):
        ltoken_total_sum += ltoken_sum_vector[field_list[i]]
        rtoken_total_sum += rtoken_sum_vector[field_list[i]]

    cdef double lrec_ave_len = ltoken_total_sum * 1.0 / ltoken_vector.size()
    cdef double rrec_ave_len = rtoken_total_sum * 1.0 / rtoken_vector.size()
    cdef double len_threshold = prefix_match_max_size * 1.0 * prefix_multiply_factor

    cdef cset[int] remained_fields
    for i in xrange(field_list.size()):
        remained_fields.insert(field_list[i])

    cdef heap[TopPair] topk_heap

    if use_plain:
        if lrec_ave_len >= len_threshold and rrec_ave_len >= len_threshold:
            topk_heap = new_topk_sim_join_plain(ltoken_vector, rtoken_vector, cand_set,
                                                prefix_match_max_size, output_size)
        else:
            topk_heap = original_topk_sim_join_plain(ltoken_vector, rtoken_vector, cand_set, output_size)
    else:
        if lrec_ave_len >= len_threshold and rrec_ave_len >= len_threshold:
            info += <char *>'new topk'
            printf("%s\n", info.c_str())
            if type == 0:
                topk_heap = new_topk_sim_join_record(ltoken_vector, rtoken_vector, lindex_vector, rindex_vector,
                                         cand_set, reuse_set, offset_of_field_num, prefix_match_max_size, output_size)
            elif type == 1:
                topk_heap = new_topk_sim_join_reuse(ltoken_vector, rtoken_vector, remained_fields,
                                         cand_set, reuse_set, offset_of_field_num, prefix_match_max_size, output_size)
        else:
            info += <char *>'original topk'
            printf("%s\n", info.c_str())
            if type == 0:
                topk_heap = original_topk_sim_join_record(ltoken_vector, rtoken_vector, lindex_vector, rindex_vector,
                                              cand_set, reuse_set, offset_of_field_num, output_size)
            elif type == 1:
                topk_heap = original_topk_sim_join_reuse(ltoken_vector, rtoken_vector, remained_fields,
                                             cand_set, reuse_set, offset_of_field_num, output_size)

    save_topk_list_to_file(field_list, output_path, topk_heap)

    return


cdef void save_topk_list_to_file(const vector[int]& field_list, const string& output_path,
                                 heap[TopPair] topk_heap) nogil:
    cdef string path = output_path + <char *>'topk_'
    cdef char buf[10]
    cdef int i
    for i in xrange(field_list.size()):
        sprintf(buf, "%d", field_list[i])
        if i != 0:
            path.append(<char *>'_')
        path.append(buf)
    path += <char *>'.txt'
    printf("%s\n", path.c_str())

    cdef TopPair pair
    cdef FILE* fp = fopen(path.c_str(), "w+")
    while topk_heap.size() > 0:
        pair = topk_heap.top()
        topk_heap.pop()
        fprintf(fp, "%.16f %d %d\n", pair.sim, pair.l_rec, pair.r_rec)
    fclose(fp)

    return


cdef void copy_table_and_remove_field(const vector[vector[int]]& table_vector,
                                      const vector[vector[int]]& index_vector,
                                      vector[vector[int]]& new_table_vector, int rm_field):
    cdef uint i, j
    for i in xrange(table_vector.size()):
        new_table_vector.push_back(vector[int]())
        for j in xrange(table_vector[i].size()):
            if index_vector[i][j] != rm_field:
                new_table_vector[i].push_back(table_vector[i][j])


cdef void remove_field(vector[vector[int]]& table_vector,
                       vector[vector[int]]& index_vector, int rm_field):
    cdef uint i, j
    for i in xrange(table_vector.size()):
        for j in reversed(range(table_vector[i].size())):
            if index_vector[i][j] == rm_field:
                index_vector[i].erase(index_vector[i].begin() + j)
                table_vector[i].erase(table_vector[i].begin() + j)


cdef void convert_table_to_vector(table_list, vector[vector[int]]& table_vector):
    cdef int i, j
    for i in range(len(table_list)):
        table_vector.push_back(vector[int]())
        for j in range(len(table_list[i])):
            table_vector[i].push_back(table_list[i][j])


cdef void convert_candidate_set_to_c_map(cand_set, cmap[int, cset[int]]& new_set):
    cdef int key, value
    for key in cand_set:
        if not new_set.count(key):
            new_set[key] = cset[int]()

        l = cand_set[key]
        for value in l:
            new_set[key].insert(value)


cdef int convert_py_list_to_vector(py_list, vector[int]& vector):
    for value in py_list:
        vector.push_back(value)


cdef double double_max(const double a, double b):
    if a > b:
        return a
    return b