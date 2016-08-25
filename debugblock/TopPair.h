class TopPair {
public:
    double sim;
    unsigned int l_rec;
    unsigned int r_rec;

    TopPair();
    TopPair(double similarity, unsigned int l_rec_idx, unsigned int r_rec_idx);
    ~TopPair();

     bool operator<(const TopPair &other) const;
     bool operator>(const TopPair &other) const;
};
