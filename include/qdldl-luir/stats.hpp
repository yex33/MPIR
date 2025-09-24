#ifndef STATS_H
#define STATS_H

#include <string>

class Stats {
public:
    Stats() { init(); };
    void init();

    double get_total_time()
    {
        return time_sym_ + time_num_fact_ + time_solve_ + time_ref_;
    }
    double get_time_num_fact(){ return time_num_fact_; }
    int get_num_refinement_steps() { return num_refinement_steps_; }
    double get_residual() { return relative_residual_; }

    std::string fact_prec_, work_prec_, res_prec_;
    int n_;
    long int nnz_;
    double time_sym_, time_num_fact_, time_solve_, time_ref_;
    int num_refinement_steps_;
    unsigned long Lnz_;
    double relative_residual_;

    void print(const char* str = 0);
};

#define AC_BLACK "" //"\x1b[30m"
#define AC_RED "" // "\x1b[31m"
#define AC_GREEN "" // "\x1b[32m"
#define AC_NORMAL "" //"\x1b[m"

#define PRINT_TIME(s, d, color)                            \
    {                                                      \
        if (d < 0.1)                                      \
            printf(color "%-20s: %.2e\n" AC_NORMAL, s, d); \
        else                                               \
            printf("%-20s: %.2f\n", s, d);                 \
    }

#define PRINT_ERROR(s, d, color)                       \
    {                                                  \
        printf(color "%-20s: %.2e\n" AC_NORMAL, s, d); \
    }

#define PRINT_INT(s, i)                         \
    {                                           \
        if (i < 1000000)                        \
            printf("%-20s: %d\n", s, int(i));   \
        else                                    \
            printf("%-20s: %ld\n", s, long(i)); \
    }

#define PRINT_STRING(s) printf("%-20s\n", s);
#define PRINT_STRING2(s1, s2) printf("%-20s: %s\n", s1, s2);

inline void Stats::init()
{
    n_ = -1;
    nnz_ = -1;
    time_sym_ = -1;
    time_num_fact_ = -1;
    time_solve_ = -1;
    time_ref_ = 0;
    num_refinement_steps_ = -1;
    Lnz_ = -1;
    relative_residual_ = -1;
}

inline std::string prec_name(const std::string& prec)
{
    if (prec == "f")
        return AC_RED + std::string("single") + AC_NORMAL;
    if (prec == "d")
        return AC_RED + std::string("double") + AC_NORMAL;
    if (prec == "7dd_real")
        return AC_RED + std::string("quadruple") + AC_NORMAL;
    if (prec == "7qd_real")
        return AC_RED + std::string("octuple") + AC_NORMAL;
    return "unknown";
}

inline void Stats::print(const char* str)
{
    printf(
        "------------------------------------------------------------------\n");
    if (str) {
        PRINT_STRING2("Matrix", str);
    } else {
        PRINT_STRING("Matrix");
    }
    PRINT_INT("   size", n_);
    PRINT_INT("   nnz in triu(A)", nnz_);
    PRINT_INT("   nnz in L", Lnz_);

    printf("%-20s: %s,  %s, %s\n",
        "precisions",
        prec_name(fact_prec_).c_str(),
        prec_name(work_prec_).c_str(),
        prec_name(res_prec_).c_str());

    PRINT_ERROR("||b-A*x||/||b||", relative_residual_, AC_RED);
    PRINT_INT("Refinement steps", num_refinement_steps_)
    PRINT_TIME("TOTAL TIME", get_total_time(), AC_RED);
    PRINT_TIME("  num fact", time_num_fact_, AC_GREEN);
    PRINT_TIME("  refine", time_ref_, AC_NORMAL);
    PRINT_TIME("  solve", time_solve_, AC_NORMAL);
    PRINT_TIME("  symbolic", time_sym_, AC_NORMAL);

    printf(
        "------------------------------------------------------------------\n");
}

#endif