// Graph t-SNE: one gradient with sparse attraction and Barnes-Hut repulsion.
// Build four modes from this file: sequential, Pthreads, OpenMP, OpenCilk.
// Uses the provided AUTh-CSAL starter/data.h and starter/data.c reader.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

extern "C" {
#include "../starter/data.h"
}

#if defined(MODE_PTHREADS)
#include <pthread.h>
#elif defined(MODE_OMP)
#include <omp.h>
#elif defined(MODE_CILK)
#include <cilk/cilk.h>
#endif

using Clock = std::chrono::steady_clock;
using I64 = int64_t;
static int gThreads = 1;
static constexpr int LEAF_CAPACITY = 16;
static constexpr int FRONTIER_DEPTH = 4;
static constexpr int MAX_TREE_DEPTH = 48;

static double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

template <typename Func>
void parallelFor(I64 n, Func&& fn) {
#if defined(MODE_PTHREADS)
    if (n == 0) return;
    int nt = static_cast<int>(std::min<I64>(gThreads, n));
    struct Job { Func* fun; I64 begin, end; };
    auto worker = [](void* arg) -> void* {
        Job* job = static_cast<Job*>(arg);
        for (I64 i = job->begin; i < job->end; ++i) (*job->fun)(i);
        return nullptr;
    };
    std::vector<Job> jobs(nt);
    std::vector<pthread_t> handles(nt > 1 ? nt - 1 : 0);
    std::vector<bool> started(nt > 1 ? nt - 1 : 0, false);
    for (int t = 0; t < nt; ++t) {
        jobs[t] = Job{&fn, n * t / nt, n * (t + 1) / nt};
    }
    for (int t = 0; t < nt - 1; ++t) {
        if (pthread_create(&handles[t], nullptr, worker, &jobs[t]) == 0)
            started[t] = true;
        else
            worker(&jobs[t]);  // Safe fallback if thread creation fails.
    }
    worker(&jobs[nt - 1]);
    for (int t = 0; t < nt - 1; ++t)
        if (started[t]) pthread_join(handles[t], nullptr);
#elif defined(MODE_OMP)
    #pragma omp parallel for schedule(dynamic, 64)
    for (I64 i = 0; i < n; ++i) fn(i);
#elif defined(MODE_CILK)
    cilk_for (I64 i = 0; i < n; ++i) fn(i);
#else
    for (I64 i = 0; i < n; ++i) fn(i);
#endif
}

// Returns natural-log entropy, using max-weight scaling for stability.
static double entropy(const graph_t& W, I64 i, double beta, double maxW) {
    double sum = 0.0, sumTLogT = 0.0;
    for (I64 k = W.colptr[i]; k < W.colptr[i+1]; ++k) {
        const double t = std::pow(W.val[k] / maxW, beta);
        sum += t;
        if (t > 0.0) sumTLogT += t * std::log(t);
    }
    return sum > 0.0 ? std::log(sum) - sumTLogT / sum : 0.0;
}

static double findBeta(const graph_t& W, I64 i, double perplexity) {
    const I64 begin = W.colptr[i], end = W.colptr[i+1];
    const I64 degree = end - begin;
    if (degree <= 1 || static_cast<double>(degree) <= perplexity) return 0.0;
    double maxW = W.val[begin];
    bool allEqual = true;
    for (I64 k = begin; k < end; ++k) {
        maxW = std::max(maxW, W.val[k]);
        if (W.val[k] != W.val[begin]) allEqual = false;
    }
    if (allEqual) return 0.0;

    const double target = std::log(perplexity);
    double lo = 0.0, hi = -1.0, beta = 1.0;
    for (int iter = 0; iter < 80; ++iter) {
        double h = entropy(W, i, beta, maxW);
        if (std::abs(h - target) < 1e-5) break;
        if (h > target) {  // Too uniform: sharpen the weights.
            lo = beta;
            beta = hi < 0.0 ? beta * 2.0 : (lo + hi) * 0.5;
        } else {           // Too sharp: flatten the weights.
            hi = beta;
            beta = (lo + hi) * 0.5;
        }
    }
    return beta;
}

struct Probabilities {
    std::vector<double> beta;
    std::vector<double> P; // Same CSC ordering as W.val.
};

static Probabilities makeProbabilities(const graph_t& W, double perplexity,
                                      double& tCond, double& tSym) {
    auto start = Clock::now();
    Probabilities result;
    result.beta.resize(W.n);
    std::vector<double> cond(W.nnz, 0.0);

    parallelFor(W.n, [&](I64 i) {
        I64 a = W.colptr[i], b = W.colptr[i + 1];
        I64 d = b - a;
        if (d == 0) { result.beta[i] = 0.0; return; }
        double beta = findBeta(W, i, perplexity);
        result.beta[i] = beta;
        if (beta == 0.0) {
            for (I64 k = a; k < b; ++k) cond[k] = 1.0 / d;
            return;
        }
        double maxW = W.val[a];
        for (I64 k = a + 1; k < b; ++k) maxW = std::max(maxW, W.val[k]);
        double sum = 0.0;
        for (I64 k = a; k < b; ++k) {
            cond[k] = std::pow(W.val[k] / maxW, beta);
            sum += cond[k];
        }
        for (I64 k = a; k < b; ++k) cond[k] /= sum;
    });
    tCond = seconds(start);

    start = Clock::now();
    result.P.resize(W.nnz);
    // CSC columns contain sorted row indices, and W is symmetric.
    parallelFor(W.n, [&](I64 i) {
        for (I64 k = W.colptr[i]; k < W.colptr[i+1]; ++k) {
            I64 j = W.row[k];
            const I64* first = W.row + W.colptr[j];
            const I64* last  = W.row + W.colptr[j+1];
            const I64* it = std::lower_bound(first, last, i);
            // Given W's documented symmetric, sorted CSC structure, it exists.
            I64 reverse = static_cast<I64>(it - W.row);
            result.P[k] = (cond[k] + cond[reverse]) / (2.0 * W.n);
        }
    });
    tSym = seconds(start);
    return result;
}

static std::vector<double> attraction(const graph_t& W, const double* Y,
                                     const std::vector<double>& P) {
    std::vector<double> attr(2 * W.n, 0.0);
    parallelFor(W.n, [&](I64 i) {
        const double xi = Y[2*i], yi = Y[2*i+1];
        double ax = 0.0, ay = 0.0;
        for (I64 k = W.colptr[i]; k < W.colptr[i+1]; ++k) {
            I64 j = W.row[k];
            double dx = xi - Y[2*j], dy = yi - Y[2*j+1];
            double a = P[k] / (1.0 + dx*dx + dy*dy);
            ax += a * dx;
            ay += a * dy;
        }
        attr[2*i] = ax;
        attr[2*i+1] = ay;
    });
    return attr;
}

// Adaptive quadtree. 'cx,cy' is geometric box center, 'mx,my' is mass center.
struct Quad {
    double cx = 0.0, cy = 0.0, half = 0.0;
    double mx = 0.0, my = 0.0;
    I64 count = 0;
    bool frontierRoot = false;
    std::vector<I64> points;                 // Only for leaves.
    std::array<std::unique_ptr<Quad>,4> child;
};

static int quadrant(const double* Y, I64 i, double cx, double cy) {
    return int(Y[2*i] >= cx) + 2 * int(Y[2*i+1] >= cy);
}

static void makeChildren(Quad& node, const double* Y,
                         std::vector<I64> ids,
                         std::array<std::vector<I64>,4>& groups) {
    for (I64 id : ids) groups[quadrant(Y, id, node.cx, node.cy)].push_back(id);
    double h = node.half * 0.5;
    for (int q = 0; q < 4; ++q) {
        if (groups[q].empty()) continue;
        node.child[q] = std::make_unique<Quad>();
        node.child[q]->cx = node.cx + ((q & 1) ? h : -h);
        node.child[q]->cy = node.cy + ((q & 2) ? h : -h);
        node.child[q]->half = h;
        node.child[q]->count = static_cast<I64>(groups[q].size());
    }
}

// Serial recursive builder for an independent subtree.
static void fillSubtree(Quad& node, const double* Y,
                        std::vector<I64> ids, int depth) {
    node.count = static_cast<I64>(ids.size());
    if (ids.size() <= LEAF_CAPACITY || depth >= MAX_TREE_DEPTH || node.half < 1e-12) {
        node.points = std::move(ids);
        return;
    }
    std::array<std::vector<I64>,4> groups;
    makeChildren(node, Y, std::move(ids), groups);
    for (int q = 0; q < 4; ++q)
        if (node.child[q]) fillSubtree(*node.child[q], Y, std::move(groups[q]), depth+1);
}

struct FrontierJob {
    Quad* node;
    std::vector<I64> ids;
    int depth;
};

// Build a shallow skeleton; its independent subtrees will be built in parallel.
static void makeFrontier(Quad& node, const double* Y,
                         std::vector<I64> ids, int depth,
                         std::vector<FrontierJob>& jobs) {
    node.count = static_cast<I64>(ids.size());
    if (depth >= FRONTIER_DEPTH || ids.size() <= LEAF_CAPACITY || node.half < 1e-12) {
        node.frontierRoot = true;
        jobs.push_back(FrontierJob{&node, std::move(ids), depth});
        return;
    }
    std::array<std::vector<I64>,4> groups;
    makeChildren(node, Y, std::move(ids), groups);
    for (int q = 0; q < 4; ++q)
        if (node.child[q]) makeFrontier(*node.child[q], Y, std::move(groups[q]), depth+1, jobs);
}

static std::unique_ptr<Quad> buildTree(const double* Y, I64 n,
                                      std::vector<FrontierJob>& jobs) {
    if (n <= 0) throw std::runtime_error("Empty point set");
    double minX=Y[0], maxX=Y[0], minY=Y[1], maxY=Y[1];
    for (I64 i=1; i<n; ++i) {
        minX = std::min(minX, Y[2*i]); maxX = std::max(maxX, Y[2*i]);
        minY = std::min(minY, Y[2*i+1]); maxY = std::max(maxY, Y[2*i+1]);
    }
    auto root = std::make_unique<Quad>();
    root->cx = minX*0.5 + maxX*0.5;
    root->cy = minY*0.5 + maxY*0.5;
    double side = std::max(maxX-minX, maxY-minY);
    root->half = 0.5 * (side*1.000001 + 1e-9);
    std::vector<I64> ids(n);
    std::iota(ids.begin(), ids.end(), I64{0});
    makeFrontier(*root, Y, std::move(ids), 0, jobs);
    parallelFor(static_cast<I64>(jobs.size()), [&](I64 j) {
        fillSubtree(*jobs[j].node, Y, std::move(jobs[j].ids), jobs[j].depth);
    });
    return root;
}

static void computeMass(Quad& node, const double* Y) {
    double sx=0.0, sy=0.0;
    if (!node.points.empty()) {
        for (I64 i : node.points) { sx += Y[2*i]; sy += Y[2*i+1]; }
    } else {
        for (auto& ch : node.child) if (ch) {
            computeMass(*ch, Y);
            sx += ch->count * ch->mx;
            sy += ch->count * ch->my;
        }
    }
    node.mx = sx / node.count;
    node.my = sy / node.count;
}

// Only merge the skeleton. All frontier subtrees already have their mass.
static void mergeSkeletonMass(Quad& node) {
    if (node.frontierRoot) return;
    double sx=0.0, sy=0.0;
    for (auto& ch : node.child) if (ch) {
        mergeSkeletonMass(*ch);
        sx += ch->count * ch->mx;
        sy += ch->count * ch->my;
    }
    node.mx = sx / node.count;
    node.my = sy / node.count;
}

struct Acc { double fx=0.0, fy=0.0, z=0.0; };

static void barnesHutWalk(const Quad& node, const double* Y, I64 i,
                          double theta, Acc& a) {
    const double xi = Y[2*i], yi = Y[2*i+1];
    if (!node.points.empty()) {
        for (I64 j : node.points) {
            if (j == i) continue; // CRITICAL: no self-interaction.
            double dx=xi-Y[2*j], dy=yi-Y[2*j+1];
            double q=1.0/(1.0+dx*dx+dy*dy);
            a.fx += q*q*dx;
            a.fy += q*q*dy;
            a.z += q;
        }
        return;
    }

    const double dx=xi-node.mx, dy=yi-node.my;
    const double dist2=dx*dx+dy*dy;
    // Never approximate a cell containing the point being evaluated.
    const bool contains = std::abs(xi-node.cx) <= node.half &&
                          std::abs(yi-node.cy) <= node.half;
    // The assignment explicitly defines r_cell as the FULL cell diagonal.
    const double diag2 = 8.0 * node.half * node.half;
    if (!contains && theta > 0.0 && dist2 > 0.0 &&
        diag2 < theta*theta*dist2) {
        double q = 1.0/(1.0+dist2);
        a.fx += node.count * q*q*dx;
        a.fy += node.count * q*q*dy;
        a.z += node.count * q;
        return;
    }
    for (const auto& ch : node.child)
        if (ch) barnesHutWalk(*ch, Y, i, theta, a);
}

struct Repulsion {
    std::vector<double> F;
    double Z=0.0;
};

static Repulsion repulsiveBH(const Quad& root, const double* Y, I64 n,
                            double theta) {
    Repulsion r;
    r.F.resize(2*n);
    std::vector<double> zi(n, 0.0);
    parallelFor(n, [&](I64 i) {
        Acc a;
        barnesHutWalk(root, Y, i, theta, a);
        r.F[2*i] = a.fx;
        r.F[2*i+1] = a.fy;
        zi[i] = a.z;
    });
    r.Z = std::accumulate(zi.begin(), zi.end(), 0.0);
    if (!(r.Z > 0.0)) throw std::runtime_error("Z must be positive");
    parallelFor(n, [&](I64 i) {
        r.F[2*i] /= r.Z;
        r.F[2*i+1] /= r.Z;
    });
    return r;
}

static Repulsion repulsiveExact(const double* Y, I64 n) {
    Repulsion r;
    r.F.resize(2*n);
    std::vector<double> zi(n, 0.0);
    parallelFor(n, [&](I64 i) {
        double fx=0, fy=0, z=0;
        for (I64 j=0; j<n; ++j) {
            if (i == j) continue;
            double dx=Y[2*i]-Y[2*j], dy=Y[2*i+1]-Y[2*j+1];
            double q=1.0/(1.0+dx*dx+dy*dy);
            fx += q*q*dx;
            fy += q*q*dy;
            z += q;
        }
        r.F[2*i] = fx;
        r.F[2*i+1] = fy;
        zi[i] = z;
    });
    r.Z = std::accumulate(zi.begin(), zi.end(), 0.0);
    if (r.Z > 0.0) parallelFor(n, [&](I64 i) {
        r.F[2*i] /= r.Z;
        r.F[2*i+1] /= r.Z;
    });
    return r;
}

static double maxAbsError(const std::vector<double>& actual,
                          const double* reference) {
    double error=0.0;
    for (size_t i=0; i<actual.size(); ++i)
        error = std::max(error, std::abs(actual[i]-reference[i]));
    return error;
}
static double relativeVectorError(const std::vector<double>& actual,
                                  const double* reference) {
    double err=0.0, scale=0.0;
    for (size_t i=0; i<actual.size(); ++i) {
        err = std::max(err, std::abs(actual[i]-reference[i]));
        scale = std::max(scale, std::abs(reference[i]));
    }
    return scale > 0.0 ? err/scale : err;
}

static void save1(const std::string& path, const std::vector<double>& x) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write: " + path);
    out << std::setprecision(17) << "node,value\n";
    for (size_t i=0; i<x.size(); ++i) out << i << ',' << x[i] << '\n';
}
static void save2(const std::string& path, const std::vector<double>& xy) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write: " + path);
    out << std::setprecision(17) << "node,x,y\n";
    for (size_t i=0; i<xy.size()/2; ++i)
        out << i << ',' << xy[2*i] << ',' << xy[2*i+1] << '\n';
}

static void benchRepulsion(const double* Y, I64 n, double theta) {
    std::vector<FrontierJob> jobs;
    auto t = Clock::now();
    auto root = buildTree(Y,n,jobs);
    double buildT = seconds(t);
    t = Clock::now();
    parallelFor(static_cast<I64>(jobs.size()), [&](I64 j) {
        computeMass(*jobs[j].node, Y);
    });
    mergeSkeletonMass(*root);
    double comT = seconds(t);
    t = Clock::now();
    Repulsion bh = repulsiveBH(*root,Y,n,theta);
    double bhT = seconds(t);
    t = Clock::now();
    Repulsion ex = repulsiveExact(Y,n);
    double exT = seconds(t);
    std::cout << "BENCH n="<<n<<" tree="<<buildT<<" com="<<comT
              <<" BH_walk="<<bhT<<" exact="<<exT
              <<" speedup(exact/BH_walk)="<<(exT/bhT)
              <<" Frep_rel_error="<<relativeVectorError(bh.F, ex.F.data())
              <<" Z_rel_error="<<std::abs(bh.Z-ex.Z)/ex.Z<<'\n';
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: "<<argv[0]<<" data/k49_norm_10NN [--threads N]"
                  << " [--theta T] [--out DIR] [--exact] [--bench-rep N]\n";
        return 1;
    }
    std::string base=argv[1];
    std::string outdir="results";
    double thetaOverride=-1.0;
    bool exact=false;
    I64 benchN=0;
    unsigned hw=std::thread::hardware_concurrency();
    gThreads=static_cast<int>(hw ? hw : 2);
#if defined(MODE_CILK)
    // OpenCilk uses this environment variable to configure its worker count.
    if (const char* workers=std::getenv("CILK_NWORKERS"))
        gThreads=std::max(1, std::atoi(workers));
#endif
    for (int a=2; a<argc; ++a) {
        std::string flag=argv[a];
        if (flag=="--threads" && a+1<argc) gThreads=std::max(1, std::stoi(argv[++a]));
        else if (flag=="--theta" && a+1<argc) thetaOverride=std::stod(argv[++a]);
        else if (flag=="--out" && a+1<argc) outdir=argv[++a];
        else if (flag=="--exact") exact=true;
        else if (flag=="--bench-rep" && a+1<argc) benchN=std::stoll(argv[++a]);
        else { std::cerr<<"Unknown argument: "<<flag<<'\n'; return 1; }
    }
#if defined(MODE_OMP)
    omp_set_num_threads(gThreads);
    const char* mode="openmp";
#elif defined(MODE_PTHREADS)
    const char* mode="pthreads";
#elif defined(MODE_CILK)
    const char* mode="opencilk";
#else
    const char* mode="sequential";
#endif
    graph_t W{};
    state_t S{};
    ref_t R{};
    try {
        if (read_graph(base.c_str(), &W)) throw std::runtime_error("read_graph failed");
        if (read_state(base.c_str(), &S)) throw std::runtime_error("read_state failed");
        if (W.n != S.n) throw std::runtime_error("Inconsistent node counts");
        if (thetaOverride >= 0.0) S.theta=thetaOverride;
        if (S.theta < 0.0) throw std::runtime_error("theta must be >=0");
        std::cout << std::setprecision(10) << "Mode="<<mode<<" n="<<W.n
                  <<" nnz="<<W.nnz<<" u="<<S.u<<" theta="<<S.theta
                  <<" threads="<<gThreads<<'\n';
        // Validate input invariants before parallel lower_bound lookups.
        for (I64 i=0; i<W.n; ++i) {
            for (I64 k=W.colptr[i]; k<W.colptr[i+1]; ++k) {
                if (W.val[k]<=0.0 || W.row[k]<0 || W.row[k]>=W.n ||
                    (k>W.colptr[i] && W.row[k]<=W.row[k-1]))
                    throw std::runtime_error("W must have positive weights and sorted unique row indices");
            }
        }
        // Optional n-sweep of Barnes-Hut versus direct repulsion.
        if (benchN>0) {
            if (benchN>W.n || benchN<2) throw std::runtime_error("--bench-rep N: require 2<=N<=graph size");
            benchRepulsion(S.Y0,benchN,S.theta);
        } else {
            if (read_ref(base.c_str(), &R)) throw std::runtime_error("read_ref failed");
            if (R.n != W.n) throw std::runtime_error("Reference size mismatch");
            auto t=Clock::now();
            double tCond=0,tSym=0;
            auto probs=makeProbabilities(W,S.u,tCond,tSym);
            double tP=seconds(t);
            t=Clock::now();
            auto attr=attraction(W,S.Y0,probs.P);
            double tAttr=seconds(t);
            std::vector<FrontierJob> jobs;
            t=Clock::now();
            auto root=buildTree(S.Y0,W.n,jobs);
            double tBuild=seconds(t);
            t=Clock::now();
            parallelFor(static_cast<I64>(jobs.size()), [&](I64 j) {
                computeMass(*jobs[j].node,S.Y0);
            });
            mergeSkeletonMass(*root);
            double tCOM=seconds(t);
            t=Clock::now();
            auto rep=repulsiveBH(*root,S.Y0,W.n,S.theta);
            double tWalk=seconds(t);
            t=Clock::now();
            std::vector<double> grad(2*W.n);
            parallelFor(2*W.n, [&](I64 i) { grad[i]=4.0*(attr[i]-rep.F[i]); });
            double tGrad=seconds(t);

            double totalP=std::accumulate(probs.P.begin(),probs.P.end(),0.0);
            std::cout << "Psum="<<totalP<<" Beta[0]="<<probs.beta[0]
                      <<" Fattr[0]=("<<attr[0]<<","<<attr[1]<<")"
                      <<" Frep[0]=("<<rep.F[0]<<","<<rep.F[1]<<")"
                      <<" Z="<<rep.Z<<'\n';
            std::cout << "Beta max abs err="<<maxAbsError(probs.beta,R.beta)
                      <<" Fattr max abs err="<<maxAbsError(attr,R.Fattr)
                      <<" Frep rel max err="<<relativeVectorError(rep.F,R.Frep)
                      <<" Z rel err="<<std::abs(rep.Z-R.Z)/R.Z<<'\n';
            std::cout << "Times (s): probabilities="<<tP<<" (conditional="<<tCond
                      <<", sym="<<tSym<<") attraction="<<tAttr
                      <<" tree="<<tBuild<<" centers="<<tCOM<<" walk="<<tWalk
                      <<" gradient="<<tGrad<<'\n';

            std::filesystem::create_directories(outdir);
            save1(outdir+"/beta.csv",probs.beta);
            save2(outdir+"/Fattr.csv",attr);
            save2(outdir+"/Frep.csv",rep.F);
            save2(outdir+"/gradient.csv",grad);
            std::ofstream zfile(outdir+"/Z.txt");
            zfile << std::setprecision(17) << rep.Z << '\n';
            std::ofstream timings(outdir+"/timings.csv");
            timings << std::setprecision(17) << "phase,seconds\n"
                    << "probabilities,"<<tP<<"\n"
                    << "conditional,"<<tCond<<"\n"
                    << "symmetrization,"<<tSym<<"\n"
                    << "attraction,"<<tAttr<<"\n"
                    << "tree_build,"<<tBuild<<"\n"
                    << "center_of_mass,"<<tCOM<<"\n"
                    << "barnes_hut_walk,"<<tWalk<<"\n"
                    << "gradient,"<<tGrad<<"\n";
            if (exact) {
                t=Clock::now();
                auto direct=repulsiveExact(S.Y0,W.n);
                double tExact=seconds(t);
                std::cout << "Direct: Z="<<direct.Z<<" seconds="<<tExact
                          <<" BH relative Frep error="<<relativeVectorError(rep.F,direct.F.data())
                          <<" speedup(walk only)="<<tExact/tWalk<<'\n';
                save2(outdir+"/Frep_exact.csv",direct.F);
                timings << "exact_repulsion,"<<tExact<<"\n";
            }
            std::cout << "Saved CSV files to "<<outdir<<'\n';
        }
        free_graph(&W);
        std::free(S.Y0);
        std::free(R.beta); std::free(R.Fattr); std::free(R.Frep);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: "<<e.what()<<'\n';
        free_graph(&W);
        std::free(S.Y0);
        std::free(R.beta); std::free(R.Fattr); std::free(R.Frep);
        return 1;
    }
}
