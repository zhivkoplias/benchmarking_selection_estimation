/*

    WFABC: a Wright-Fisher ABC-based approach for inferring effective population 
    sizes and selection coefficients from time-sampled data.
    Copyright (C) 2014  Matthieu Foll

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

	Contact: matthieu.foll@epfl.ch

*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <omptl/omptl_algorithm>
#include <unistd.h>

#define STOC_BASE CRandomSFMT
#include "randomc/sfmt.cpp"             // code for random number generator
#include "randomc/sfmt.h"
#include "stocc/stocc.h"              // define random library classes
#include "stocc/stoc1.cpp"            // random library source code
#include "randomc/userintf.cpp"         // define system specific user interface

#include <cstring>
#include "anyoption/anyoption.h"
#include "anyoption/anyoption.cpp"


#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#define omp_get_max_threads() 1
#endif

#define rbinom randgen_parallel[omp_get_thread_num()].Binomial
#define runif(a,b) a+(b-a)*randgen_parallel[omp_get_thread_num()].Random()
#define modulo(a,b) ((a % b) + b) % b

using namespace std;

// http://www.concentric.net/~Ttwang/tech/inthash.htm
unsigned long mix(unsigned long a, unsigned long b, unsigned long c) {
    a = a - b; a = a - c; a = a^(c >> 13); b = b - c; b = b - a; b = b^(a << 8); c = c - a; c = c - b; c = c^(b >> 13); a = a - b; a = a - c; a = a^(c >> 12); b = b - c; b = b - a; b = b^(a << 16); c = c - a; c = c - b; c = c^(b >> 5); a = a - b; a = a - c; a = a^(c >> 3); b = b - c; b = b - a; b = b^(a << 10); c = c - a; c = c - b; c = c^(b >> 15); return c;
}

vector<double> all_dist;
bool sort_index(size_t i, size_t j) {
    return (all_dist[i] < all_dist[j]);
}

int main(int argc, char **argv) {

    unsigned int seed = 0;
    unsigned int nreps = 100000; // number of simulations
    double acc_rate = 0.01; //acceptance rate in ABC
    bool s_list_provided=false; // varying s at each simulation
    bool N_list_provided=false; // varying N at each generation
    bool N_boot=true; // varying N from bootstrap distribution
    unsigned int fixed_N; // fixed population size in number of chromosomes
    unsigned int fixed_N_sample=0; // fixed sample size (number of chromosomes)
    unsigned int fixed_j=1; // number of mutants at first generation (!!! NOT NB OF INDIVIDUALS BUT NB OF GENES !!!)
    double min_s = -0.3; // lower bound for the uniform prior on s
    double max_s = 0.6; // higher bound
    double mean_s = 0; // mean for the normal prior on s
    double sd_s = 0; // sd  for the normal prior on s (if 0 use the uniform))
    double min_h = 0.5; // dominance level
    double max_h = 0.5; // dominance level
    unsigned int max_sims = 1; // max number of simulations tried for a fixed value of s before giving up
    unsigned int ploidy = 2; // ploidy level
    double min_freq = 0.01; // minimum frequency observed
    double min_freq_last = 0.0; // minimum frequency observed at the last time point
    bool poly_condition = false; // conditioned on being polymorphic at the last time point
    double min_freq_fs = 0.01; // minimum frequency to calculate Fs'
    double freq_threshold = 0.05; // used to calculate time below this freq (used to estimate h)
    double fs_threshold = 0.00; // Minor frequency threshold to decompose Fsi and Fsd
    
    bool write_last_sims=false; // write out last simulated data. Usefull when simulating only once or when using only one locus.
    bool simulations_provided=false; // simulations are provided by the user, otherwise are redone for each locus
    bool simulate_once=false; // true if fixed_N_sample and fixed_j are provided
    
    //ofstream tempo("tempo.txt");
    //unsigned int cur_locus=1;  
    //ofstream malas("multiple_loci_malas.txt");

    AnyOption *opt = new AnyOption();

    opt->noPOSIX();
    //opt->setVerbose();

    opt->setFlag("help");
    opt->addUsage("");
    opt->setFlag("help",'h');
    opt->addUsage( " ----------------------------- " );
	opt->addUsage( " | WFABC_2 1.0 usage:        | " );
	opt->addUsage( " |  values indicate defaults | " );
    opt->addUsage( " |  n indicates integer      | " );
    opt->addUsage( " |  x indicates float        | " );
	opt->addUsage( " ----------------------------- " );
    opt->addUsage( " input.txt         Name of the input file " );
    opt->addUsage( " -help             Prints this help " );
    opt->addUsage( " -nthreads n       Number of threads (automatically detected by default) " );
    opt->addUsage( " -ploidy 2         Ploidy of individuals (use 1 or 2) " );
    opt->addUsage( " --------------------------- " );
	opt->addUsage( " | ABC options             | " );
	opt->addUsage( " --------------------------- " );
    opt->addUsage( " -nreps 100000     Total number of simulations " );
    opt->addUsage( " -acc_rate 0.01    Proportion of accepted simulations " );
    opt->addUsage( " -fixed_N n        Set Ne to a fixed value (does not estimate it) " );
    opt->addUsage( " -min_s x          Lower bound of a uniform prior for s" );
    opt->addUsage( " -max_s x          Higher bound of a uniform prior for s" );
    opt->addUsage( " -mean_s x         Mean of a Normal prior for s" );
    opt->addUsage( " -sd_s x           Standard deviation of a Normal prior for s" );
    opt->addUsage( " -s_list file_name User defined s prior (one random sample per line)" );
    opt->addUsage( " -min_h x          Lower bound of a uniform prior for h" );
    opt->addUsage( " -max_h x          Higher bound of a uniform prior for h" );
    opt->addUsage( " --------------------------- " );
	opt->addUsage( " | Conditioning             | " );
	opt->addUsage( " --------------------------- " );
    opt->addUsage( " -min_freq 0.01    Condition on a minimum frequency over all time points" );
    opt->addUsage( " -min_freq_last x  Condition on a minimum frequency for the last time point" );
    opt->addUsage( " -poly_condition   Condition on being polymorphic at the last time point" );
    opt->addUsage( " -max_sims 1       Maximum number of trials to match conditions before changing s, h and Ne" );
    opt->addUsage( " --------------------------- " );
	opt->addUsage( " | Advanced ABC options    | " );
	opt->addUsage( " --------------------------- " );
    opt->addUsage( " -fixed_N_sample n Fixed sample size (ignore those in the input file)" );
    opt->addUsage( " -fixed_j n        Fixed initial number of mutants (ignore those in the input file)" );
    opt->addUsage( " -N_list file_name List of values for Ne at each generation (one per line)" );
    opt->addUsage( " -sims file_name   Use simulations given in file_name for all loci (see wls)" );
    opt->addUsage( " -wls              Write out simulations for the last locus" );
    opt->addUsage( " ---------------------------------------------- " );
	opt->addUsage( " | Advanced options (change with caution)     | " );
	opt->addUsage( " ---------------------------------------------- " );
    opt->addUsage( " -min_freq_fs 0.01 Minimum allele frequency to calculate the Fs statistics " );
    opt->addUsage( " -F_threshold 0.00 Minor frequency threshold to decompose Fsi and Fsd" );
    opt->addUsage( " -t_threshold 0.05 Minor frequency threshold to calculate tl and th" );
    opt->addUsage( " -seed n           Manually set the seed (used for debbuging)" );
    
    
    opt->setOption("seed");
    opt->setOption("nreps");
    opt->setOption("acc_rate");
    opt->setOption("fixed_N_sample");
    opt->setOption("fixed_N");
    opt->setOption("fixed_j");
    opt->setOption("min_s");
    opt->setOption("max_s");
    opt->setOption("min_h");
    opt->setOption("max_h");
    opt->setOption("mean_s");
    opt->setOption("sd_s");
    opt->setOption("max_sims");
    opt->setOption("ploidy");
    opt->setOption("min_freq");
    opt->setOption("min_freq_last");
    opt->setOption("min_freq_fs");
    opt->setOption("s_list");
    opt->setOption("N_list");
    opt->setOption("sims");
    opt->setOption("F_threshold");
    opt->setOption("t_threshold");
    opt->setFlag("wls");
    opt->setFlag("poly_condition");
    opt->setOption("nthreads");

    opt->processCommandArgs(argc, argv);
 
    if (!opt->hasOptions() || opt->getArgc() == 0) { /* print usage if no options */
        opt->printUsage();
        delete opt;
        return 1;
    }
    if (opt->getFlag("help"))
        opt->printUsage();

    if (!((opt->getValue("sims") != NULL) || (opt->getValue("s_list") != NULL) || ((opt->getValue("min_s") != NULL) && (opt->getValue("max_s") != NULL)) || ((opt->getValue("mean_s") != NULL) && (opt->getValue("sd_s") != NULL)))) {
        cerr << "Please provide either a uniform or a normal prior for s, or a list of fixed values, or precomputed simulations." << endl;
        return 1;
    }
    if (opt->getValue("N_list") && opt->getValue("fixed_N")) {
        cerr << "Please provide only one of the two options N_list or fixed_N." << endl;
        return 1;
    }

    if (opt->getValue("seed") != NULL)
        seed = atoi(opt->getValue("seed"));
    if (opt->getValue("nreps") != NULL)
        nreps = atoi(opt->getValue("nreps"));
    if (opt->getValue("acc_rate") != NULL)
        acc_rate = atof(opt->getValue("acc_rate"));
    if (opt->getValue("fixed_N_sample") != NULL)
        fixed_N_sample = atoi(opt->getValue("fixed_N_sample"));
    if (opt->getValue("fixed_N") != NULL) {
        fixed_N = atoi(opt->getValue("fixed_N"));
        N_boot=false;
    }
    if (opt->getValue("fixed_j") != NULL)
        fixed_j = atoi(opt->getValue("fixed_j"));
    if (opt->getValue("min_s") != NULL)
        min_s = atof(opt->getValue("min_s"));
    if (opt->getValue("max_s") != NULL)
        max_s = atof(opt->getValue("max_s"));
    if (opt->getValue("mean_s") != NULL)
        mean_s = atof(opt->getValue("mean_s"));
    if (opt->getValue("sd_s") != NULL)
        sd_s = atof(opt->getValue("sd_s"));
    if (opt->getValue("min_h") != NULL)
        min_h = atof(opt->getValue("min_h"));
    if (opt->getValue("max_h") != NULL)
        max_h = atof(opt->getValue("max_h"));
    if (opt->getValue("max_sims") != NULL)
        max_sims = atoi(opt->getValue("max_sims"));
    if (opt->getValue("ploidy") != NULL)
        ploidy = atoi(opt->getValue("ploidy"));
    if (opt->getValue("min_freq") != NULL)
        min_freq = atof(opt->getValue("min_freq"));
    if (opt->getValue("min_freq_last") != NULL)
        min_freq_last = atof(opt->getValue("min_freq_last"));
    if (opt->getValue("min_freq_fs") != NULL)
        min_freq_fs = atof(opt->getValue("min_freq_fs"));
    if (opt->getValue("t_threshold") != NULL)
        freq_threshold = atof(opt->getValue("t_threshold"));
    if (opt->getValue("F_threshold") != NULL)
        fs_threshold = atof(opt->getValue("F_threshold"));
    if (opt->getFlag("poly_condition"))
        poly_condition = true;
    if (opt->getFlag("wls"))
        write_last_sims = true;
    if (opt->getValue("fixed_N_sample") != NULL &&  opt->getValue("fixed_j") != NULL) {
        simulate_once=true;
    }
    // list of s provided
    vector<double> user_s;
    if (opt->getValue("s_list")) {
        user_s.reserve(nreps);
        s_list_provided=true;
        string s_list_name=opt->getValue("s_list");
        ifstream list_s(s_list_name.c_str());
        double tmp_s;
        while (list_s >> tmp_s) {
            user_s.push_back(tmp_s);
        }
    }
    // fixed N at each generation
    if (opt->getValue("N_list")) {
        N_list_provided=true;
        N_boot=false;
    }
    
    // read data
    string file_name=opt->getArgv(0);
    
    string file_prefix = file_name;
    if (file_prefix.find(".", 1) != string::npos)
        file_prefix = file_prefix.substr(0, file_prefix.length() - 4);
    //unsigned long position;
    //if ((position = file_prefix.find_last_of('/')))
    //    file_prefix = file_prefix.substr(position + 1, file_prefix.length());
 
    vector<unsigned int> boot_N;
    // read N values from bootstrap distribution
    if (N_boot) {
        boot_N.reserve(nreps);
        string N_boot_name=file_prefix + "_Ne_bootstrap.txt";
        ifstream boot_N_file(N_boot_name.c_str());
        double tmp_N;
        while (boot_N_file >> tmp_N) {
            boot_N.push_back((int)round(tmp_N));
        }
    }
    
    // simulations
    vector<double> res_s; // simulated s values
    vector<double> res_h; // simulated s values
    vector<double> res_fsd; // simulated fsd
    vector<double> res_fsi; // simulated fsi
    vector<double> res_fsd_l; // simulated fsd
    vector<double> res_fsi_l; // simulated fsi
    vector<double> res_tl; // simulated tl (time lower than freq_threshold)
    vector<double> res_th; // simulated th (time higher than 1-freq_threshold)
    vector<unsigned int> res_N; // simulated N
    res_s.reserve(nreps);
    res_h.reserve(nreps);
    res_fsd.reserve(nreps);
    res_fsi.reserve(nreps);
    res_fsd_l.reserve(nreps);
    res_fsi_l.reserve(nreps);
    res_tl.reserve(nreps);
    res_th.reserve(nreps);
    res_N.reserve(nreps);
    // provided simulations
    if (opt->getValue("sims")) {
        simulations_provided=true;
        string sim_name=opt->getValue("sims");
        ifstream sim_file(sim_name.c_str());
        string sim_line;
        nreps=0;
        // skip header line
        getline(sim_file,sim_line);
        double tmp_d;
        unsigned int tmp_i;
        while (getline(sim_file,sim_line)) {
            nreps++;
            istringstream sims_line(sim_line);
            sims_line >> tmp_d;
            res_fsi.push_back(tmp_d);
            sims_line >> tmp_d;
            res_fsd.push_back(tmp_d);
            sims_line >> tmp_d;
            res_fsi_l.push_back(tmp_d);
            sims_line >> tmp_d;
            res_fsd_l.push_back(tmp_d);
            sims_line >> tmp_d;
            res_tl.push_back(tmp_d);
            sims_line >> tmp_d;
            res_th.push_back(tmp_d);
            sims_line >> tmp_d;
            res_s.push_back(tmp_d);
            sims_line >> tmp_d;
            res_h.push_back(tmp_d);
            sims_line >> tmp_i;
            res_N.push_back(tmp_i);
        }

    } else {
        res_s.resize(nreps);
        res_h.resize(nreps);
        res_fsd.resize(nreps);
        res_fsi.resize(nreps);
        res_fsd_l.resize(nreps);
        res_fsi_l.resize(nreps);
        res_tl.resize(nreps);
        res_th.resize(nreps);
        res_N.resize(nreps);
    }
    
    unsigned int nloci;
    unsigned int nb_times;
    unsigned int t; // number of generations to simulate
    // read number of loci and number of times
    string line;
    // read intput file
    ifstream infile(file_name.c_str());
    getline (infile,line);
    istringstream header_line(line);
    header_line >> nloci;
    header_line >> nb_times;
    // read time points
    getline (infile,line);
    istringstream time_line(line);
    vector<int> sample_times(nb_times); // time of the samples (in generations)
    int cur_time;
    for (unsigned int i = 0; i < nb_times; ++i) {
        time_line >> cur_time;
        sample_times[i]=cur_time;
        time_line.ignore(1);
    }
    // total number of generations
    t=sample_times.back()-sample_times.front()+1;

    // fixed N at each generation
    vector<unsigned int> user_N(t);
    if (N_list_provided) {
        string N_list_name=opt->getValue("N_list");
        ifstream list_N(N_list_name.c_str());
        for (unsigned int k = 0; k < t; ++k) {
            list_N >> user_N[k];
        }
    }
    
    // create and seed parallels randgens
    int nthreads = omp_get_max_threads();
    if (opt->getValue("nthreads") != NULL) {
        #ifdef _OPENMP
            nthreads = atoi(opt->getValue("nthreads"));
        #else
            cerr << "Warning: this version does not support multithreading, nthreads option will be ignored" << endl;
        #endif
    }
    #ifdef _OPENMP
        cerr << "Using " << nthreads << " thread";
        if (nthreads > 1) cerr << "s";
        cerr << endl;
        omp_set_num_threads(nthreads);
    #endif
    vector<StochasticLib1> randgen_parallel(nthreads, StochasticLib1(0));
    int seeds[2];
    if (seed == 0) {
        seeds[0] = (int)mix(clock(), time(NULL), getpid());
    } else {
        seeds[0] = seed;
    }
    for (int k = 0; k < nthreads; k++) {
        seeds[1] = k;
        randgen_parallel[k].RandomInitByArray(seeds, 2);
    }

    // write output for tempoFS
    //tempo << nb_times << " " << nreps << endl;
    //for (unsigned int rep = 0; rep < nreps; ++rep)
    //   tempo << 2 << " ";
    //tempo << endl;

    int cur_sample;
    int init_N_sample;
    unsigned int j;
    double init_freq;
    unsigned int mutation_start; // time at which the first mutant is detected
    vector<unsigned int> N_sample(nb_times); // sample sizes at each time point (nb of individuals)
    ifstream obs_file((file_prefix + "_obs_stats.txt").c_str());
    getline (obs_file,line);
    ofstream s_out((file_prefix + "_posterior_s.txt").c_str());
    ofstream h_out;
    if (min_h!=max_h || simulations_provided) h_out.open((file_prefix + "_posterior_h.txt").c_str());
    ofstream N_out;
    if (N_boot) N_out.open(((file_prefix + "_posterior_N.txt").c_str()));
    // loop over loci
    for (unsigned int l = 0; l < nloci; ++l) {
        cerr << "Locus " << l+1 << endl;
        // perform simulations if not provided and if
        if (!(simulations_provided || (simulate_once && l!=0))) {
            if (simulate_once) {
                for (unsigned int i = 0; i < nb_times; ++i)
                    N_sample[i]=fixed_N_sample;
                j=fixed_j;
                mutation_start=0;
            }
            else {
                getline (infile,line);
                istringstream cur_line_N_sample(line);
                // read sample sizes
                int cur_N_sample;
                for (unsigned int i = 0; i < nb_times; ++i) {
                    cur_line_N_sample >> cur_N_sample;
                    N_sample[i]=cur_N_sample;
                    cur_line_N_sample.ignore(1);
                }
                // find first time with a non-zero number of alleles
                getline (infile,line);
                istringstream cur_line_sample(line);
                //int cur_sample;
                mutation_start=-1;
                do {
                    cur_line_sample >> cur_sample;
                    mutation_start++;
                    cur_line_sample.ignore(1);
                }
                while (cur_sample==0);
                init_freq=(double)cur_sample/(double)N_sample[mutation_start];
                init_N_sample=N_sample[mutation_start];
                // transform in number of generations
                mutation_start=sample_times[mutation_start]-sample_times.front();
            }
            #pragma omp parallel
            {
                vector<unsigned int> x(t);
                vector<unsigned int> sample_count(nb_times);
                vector<double> sample_freq(nb_times);
                vector<double> fsp(nb_times - 1);
                double x_fs, y_fs, z_fs, fsi, fsd, fsi_l, fsd_l, tl, th, nt_fs;
                unsigned int ngen_fs,ngen_fs_l;
                double max_sample_freq;
                double wAA, wAa, waa;
                double s,h;
                unsigned int N;
                double p;
                double prob;
                unsigned int nsims;
                #pragma omp for firstprivate(j)
                for (unsigned int rep = 0; rep < nreps; ++rep) {
                    nsims = 0;
                    if (s_list_provided) {
                        s = user_s[modulo(rep,user_s.size())];
                    }
                    else {
                        if (sd_s > 0) {
                            s = randgen_parallel[omp_get_thread_num()].Normal(mean_s, sd_s);
                        } else {
                            s = runif(min_s, max_s);
                        }
                    }
                    h = runif(min_h, max_h);
                    if (!N_list_provided) {
                        if (N_boot) {
                            N=boot_N[modulo(rep,boot_N.size())];
                        } else {
                            N = fixed_N;
                        }
                        if (N < 10) {
                            #pragma omp critical
                            {
                                cerr << "Warning, value of N=" << N << " has been replaced by 10" << endl;
                            }
                            N = 10;
                        }
                    }
                    else {
                        N=user_N[0];
                    }
                    if (!simulate_once) {
                        // http://en.wikipedia.org/wiki/Beta_distribution#Rule_of_succession
                        // http://en.wikipedia.org/wiki/Beta_distribution#Generating_beta-distributed_random_variates
                        // http://en.wikipedia.org/wiki/Gamma_distribution#Generating_gamma-distributed_random_variables
                        // generate beta(cur_sample+1,N_sample[mutation_start]-cur_sample+1)
                        // this is equivalent to generate x/(x+y) with x~gamma(cur_sample+1,1) and y~gamma(N_sample[mutation_start]-cur_sample+1,1)
                        // x~gamma(cur_sample+1,1) is equivalent to generate sum(-ln(U_k),k=1..cur_sample+1)
                        // y~gamma(N_sample[mutation_start]-cur_sample+1,1) is equivalent to generate sum(-ln(U_k),k=1..N_sample[mutation_start]-cur_sample+1)
                        double x_tmp=0;
                        for (int k_tmp=0;k_tmp<cur_sample+1;++k_tmp) x_tmp+=-log(runif(0,1));
                        double y_tmp=0;
                        for (int k_tmp=0;k_tmp<init_N_sample-cur_sample+1;++k_tmp) y_tmp+=-log(runif(0,1));
                        j=max(1,(int)round((x_tmp*(double)N)/(x_tmp+y_tmp)));
                        //j=max(1,(int)round(init_freq*(double)N));
                    }
                    while (1) {
                        nsims++;
                        for (unsigned int k = 0; k < mutation_start; ++k) {
                            x[k] = 0;
                        }
                        x[mutation_start] = j;
                        // WF simulation
                        if (N_list_provided) {
                            N=user_N[mutation_start];
                        }
                        for (unsigned int k = mutation_start+1; k < t; ++k) {
                            // absorbing states
                            if (x[k - 1] == 0) {
                                if (N_list_provided) {
                                    N=user_N[k];
                                }
                                x[k] = 0;
                            }
                            else if (x[k - 1] == N) {
                                if (N_list_provided) {
                                    N=user_N[k];
                                }
                                x[k] = N;
                            }
                            else {
                                p = (double) x[k - 1] / (double) (N);
                                wAA = 1 + s;//(-1+2*exp(-s*p)); (-1+1.3*exp(-s*p))
                                wAa = 1 + s*h;//(-1+2*exp(-s*p))*h;
                                waa = 1;
                                // calculate sampling probabilities
                                if (ploidy == 2) {
                                    prob = (wAA * p * p + wAa * p * (1 - p)) / (wAA * p * p + wAa * 2 * p * (1 - p) + waa * (1 - p)* (1 - p));
                                } else {
                                    prob = wAA * p / (wAA * p + waa * (1 - p));
                                }
                                if (N_list_provided) {
                                    N=user_N[k];
                                }
                                if (poly_condition) {
                                    do {
                                        x[k] = rbinom(N, prob);
                                    } while (x[k] == 0 || x[k] == N);
                                } else if (min_freq_last > 0) {
                                    do {
                                        x[k] = rbinom(N, prob);
                                    } while (x[k] == 0);
                                } else {
                                    x[k] = rbinom(N, prob);
                                }
                            }
                        }
                        // time sampling
                        max_sample_freq = 0;
                        for (unsigned int i = 0; i < nb_times; ++i) {
                            if (N_list_provided) {
                                N=user_N[sample_times[i] - sample_times.front()];
                            }
                            sample_count[i] = rbinom(N_sample[i], (double) x[sample_times[i] - sample_times.front()] / (double) (N));
                            sample_freq[i] = (double) sample_count[i] / (double) (N_sample[i]);
                            if (sample_freq[i] > max_sample_freq) max_sample_freq = sample_freq[i];
                        }
                        
                        // calculate Fs'i and Fs'd
                        ngen_fs = 0;
                        ngen_fs_l = 0;
                        fsi = 0;
                        fsd = 0;
                        fsi_l = 0;
                        fsd_l = 0;
                        for (unsigned int i = 1; i < nb_times; ++i) {
                            x_fs = sample_freq[i - 1];
                            y_fs = sample_freq[i];
                            if (y_fs !=0 && y_fs < freq_threshold)
                                tl=tl+1;
                            if (y_fs !=1 && y_fs > 1-freq_threshold)
                                th=th+1;
                            // check the mini frequency condition and calculate Fs' if possible
                            if (!(((x_fs < min_freq_fs) & (y_fs < min_freq_fs)) || (((1 - x_fs) < min_freq_fs) & ((1 - y_fs) < min_freq_fs)))) {
                                z_fs = (x_fs + y_fs) / 2;
                                fsp[i - 1] = (x_fs - y_fs) * (x_fs - y_fs) / (z_fs * (1 - z_fs));
                                nt_fs = 1 / ((1 / (double) (N_sample[i - 1]) + 1 / (double) (N_sample[i])) / 2);
                                fsp[i - 1] = ((fsp[i - 1]*(1 - 1 / (2 * nt_fs)) - 2 / nt_fs) / ((1 + fsp[i - 1] / 4)*(1 - 1 / ((double) (N_sample[i])))));
                                // calculate the total number of generation where we could calculate Fs'
                                if ( ((x_fs < fs_threshold) & (y_fs < fs_threshold)) || (((1 - x_fs) < fs_threshold) & ((1 - y_fs) < fs_threshold)) ) {
                                    ngen_fs_l = ngen_fs_l + (sample_times[i] - sample_times[i - 1]);
                                    if (x_fs < y_fs)
                                        fsi_l = fsi_l + fsp[i - 1];
                                    else //if (x_fs > y_fs)
                                        fsd_l = fsd_l + fsp[i - 1];
                                } else {
                                    ngen_fs = ngen_fs + (sample_times[i] - sample_times[i - 1]);
                                    if (x_fs < y_fs)
                                        fsi = fsi + fsp[i - 1];
                                    else //if (x_fs > y_fs)
                                        fsd = fsd + fsp[i - 1];
                                }
                            }
                        }
                        if (ngen_fs > 0) {
                            fsi = fsi / ngen_fs;
                            fsd = fsd / ngen_fs;
                        }
                        if (ngen_fs_l > 0) {
                            fsi_l = fsi_l / ngen_fs_l;
                            fsd_l = fsd_l / ngen_fs_l;
                        }
                        unsigned int first_non_fixed=0;
                        while ((sample_freq[first_non_fixed]==0 || sample_freq[first_non_fixed]==1) && first_non_fixed!=nb_times-1) {
                            first_non_fixed++;
                        }
                        unsigned int last_non_fixed=nb_times-1;
                        while ((sample_freq[last_non_fixed]==0 || sample_freq[last_non_fixed]==1) && last_non_fixed!=0) {
                            last_non_fixed--;
                        }
                        tl=0;
                        th=0;
                        /*for (unsigned int i = first_non_fixed; i < last_non_fixed+1; ++i) {
                             if (sample_freq[i] < freq_threshold)
                                 tl=tl+1;
                             if (sample_freq[i] > 1-freq_threshold)
                                 th=th+1;
                        }*/
                        for (unsigned int i = max((int)first_non_fixed,1); i < min(last_non_fixed+1,nb_times); ++i) {
                            double x1=sample_times[i-1];
                            double x2=sample_times[i];
                            double y1=sample_freq[i-1];
                            double y2=sample_freq[i];
                            if (y2!=y1) { // do linear interpolation to count the number of generations
                                double x3_l=x1+(x2-x1)*(freq_threshold-y1)/(y2-y1);
                                double x3_h=x1+(x2-x1)*(1-freq_threshold-y1)/(y2-y1);
                                if (y1<y2) {
                                    tl=tl+max(min(x3_l-x1,x2-x1),0.0);
                                    th=th+max(min(x2-x3_h,x2-x1),0.0);
                                } else {
                                    tl=tl+max(min(x2-x3_l,x2-x1),0.0);
                                    th=th+max(min(x3_h-x1,x2-x1),0.0);
                                }
                            }
                            else { // if both freq are equal
                                if (y1<freq_threshold) tl=tl+x2-x1;
                                if (y1>1-freq_threshold) th=th+x2-x1;
                            }
                        }
                        
                        // conditions to stop simulations
                        if (ngen_fs+ngen_fs_l > 0 && max_sample_freq >= min_freq && (double) sample_count[nb_times - 1] / (double) (N_sample[nb_times - 1]) >= min_freq_last) {
                            if (poly_condition) {
                                if ((sample_count[nb_times - 1] != 0) & (sample_count[nb_times - 1] != N_sample[nb_times - 1])) {
                                    break;
                                }
                            } else {
                                break;
                            }
                        }
                        
                        // change s and Ne if we already tried too many times
                        if (nsims >= max_sims && max_sims>1) {
                            #pragma omp critical
                            {
                                cerr << "Warning, max number of simulations tried for locus " << l+1 << " with s=" << s << endl;
                            }
                            if (s_list_provided) {
                                s = user_s[modulo(rep,user_s.size())];
                            }
                            else {
                                if (sd_s > 0) {
                                    s = randgen_parallel[omp_get_thread_num()].Normal(mean_s, sd_s);
                                } else {
                                    s = runif(min_s, max_s);
                                }
                            }
                            h = runif(min_h, max_h);
                            if (!N_list_provided) {
                                if (N_boot) {
                                    N = boot_N[modulo(rep,boot_N.size())];
                                } else {
                                    N = fixed_N;
                                }
                                if (N < 10) {
                                    cerr << "Warning, value of N=" << N << " has been replaced by 10" << endl;
                                    N = 10;
                                }
                            }
                            nsims = 0;
                        }
                        
                    } // end of one successful simulation
                    res_s[rep] = s;
                    res_h[rep] = h;
                    res_fsi[rep] = fsi;
                    res_fsd[rep] = fsd;
                    res_fsi_l[rep] = fsi_l;
                    res_fsd_l[rep] = fsd_l;
                    res_tl[rep] = tl;
                    res_th[rep] = th;
                    res_N[rep] = N;
                    // BE CAREFUL HERE, THE ORDER OF THE OUTPUT IS NOT THE SAME AS THE MAIN OUTPUT UNLESS NUM_THREADS=1
                    /*#pragma omp critical
                     {
                     // write output for tempoFS
                     //for (unsigned int i = 0; i < nb_times; ++i) {
                     //    double cur_freq=((double)sample_count[i])/((double) (N_sample[i]));
                     //    tempo << cur_freq << " " << 1-cur_freq << " " << N_sample[i] << " locus" << cur_locus << " " << sample_times[i] <<endl;
                     //}
                     //cur_locus++;
                     
                     // write output for Malaspinas
                     for (unsigned int i = 0; i < nb_times; ++i)
                     {
                     malas << (int)sample_times[i]-(int)t;
                     if (i<nb_times-1) malas << ",";
                     }       
                     malas << endl;
                     for (unsigned int i = 0; i < nb_times; ++i)
                     {
                     malas << N_sample[i];
                     if (i<nb_times-1) malas << ",";
                     }
                     malas << endl;
                     for (unsigned int i = 0; i < nb_times; ++i)
                     {
                     malas << sample_count[i];
                     if (i<nb_times-1) malas << ",";
                     }
                     malas << endl;
                     }
                     */
                }
            }// end of the parallel region
        } // end of simulation
        // rejection ABC step for this locus
        double obs_Fsi;
        double obs_Fsd;
        double obs_Fsi_l;
        double obs_Fsd_l;
        double obs_tl;
        double obs_th;
        // read observed stats
        getline (obs_file,line);
        istringstream obs_line(line);
        obs_line >> obs_Fsi;
        obs_line >> obs_Fsd;
        obs_line >> obs_Fsi_l;
        obs_line >> obs_Fsd_l;
        obs_line >> obs_tl;
        obs_line >> obs_th;
        // calculate sd of stats
        double mean_fsi=0,sd_fsi=0;
        double mean_fsd=0,sd_fsd=0;
        double mean_fsi_l=0,sd_fsi_l=0;
        double mean_fsd_l=0,sd_fsd_l=0;
        double mean_tl=0,sd_tl=0;
        double mean_th=0,sd_th=0;
        for (unsigned int rep = 0; rep < nreps; ++rep) {
            mean_fsi+=res_fsi[rep];
            sd_fsi+=res_fsi[rep]*res_fsi[rep];
            mean_fsd+=res_fsd[rep];
            sd_fsd+=res_fsd[rep]*res_fsd[rep];
            mean_fsi_l+=res_fsi_l[rep];
            sd_fsi_l+=res_fsi_l[rep]*res_fsi_l[rep];
            mean_fsd_l+=res_fsd_l[rep];
            sd_fsd_l+=res_fsd_l[rep]*res_fsd_l[rep];
            mean_tl+=res_tl[rep];
            sd_tl+=res_tl[rep]*res_tl[rep];
            mean_th+=res_th[rep];
            sd_th+=res_th[rep]*res_th[rep];
        }
        mean_fsi=mean_fsi/nreps;
        sd_fsi=sqrt(sd_fsi / nreps - mean_fsi * mean_fsi);
        mean_fsd=mean_fsd/nreps;
        sd_fsd=sqrt(sd_fsd / nreps - mean_fsd * mean_fsd);
        mean_fsi_l=mean_fsi_l/nreps;
        sd_fsi_l=sqrt(sd_fsi_l / nreps - mean_fsi_l * mean_fsi_l);
        mean_fsd_l=mean_fsd_l/nreps;
        sd_fsd_l=sqrt(sd_fsd_l / nreps - mean_fsd_l * mean_fsd_l);
        mean_tl=mean_tl/nreps;
        sd_tl=sqrt(sd_tl / nreps - mean_tl * mean_tl);
        mean_th=mean_th/nreps;
        sd_th=sqrt(sd_th / nreps - mean_th * mean_th);
 
        // normalize both Fsi and Fsd by the largest sd
        double sd_fs=max(max(sd_fsi,sd_fsd),max(sd_fsi_l,sd_fsd_l));
        double sd_thl=max(sd_th,sd_tl);

        // calculate distances
        all_dist.resize(nreps);
        #pragma omp for
        for (unsigned int rep = 0; rep < nreps; ++rep) {
            all_dist[rep]= (res_tl[rep]/sd_thl-obs_tl/sd_thl)*(res_tl[rep]/sd_thl-obs_tl/sd_thl) + (res_th[rep]/sd_thl-obs_th/sd_thl)*(res_th[rep]/sd_thl-obs_th/sd_thl)
                            + (res_fsi[rep]/sd_fs-obs_Fsi/sd_fs)*(res_fsi[rep]/sd_fs-obs_Fsi/sd_fs) + (res_fsd[rep]/sd_fs-obs_Fsd/sd_fs)*(res_fsd[rep]/sd_fs-obs_Fsd/sd_fs)
                            + (res_fsi_l[rep]/sd_fs-obs_Fsi_l/sd_fs)*(res_fsi_l[rep]/sd_fs-obs_Fsi_l/sd_fs) + (res_fsd_l[rep]/sd_fs-obs_Fsd_l/sd_fs)*(res_fsd_l[rep]/sd_fs-obs_Fsd_l/sd_fs);
        }
       // sort and keep track of indices of elements
        vector<int> indices(nreps);
        //  vector<int> indices_h(nreps);
        for (unsigned int rep = 0; rep < nreps; ++rep) {
            indices[rep]=rep;
        }
        omptl::sort(indices.begin(), indices.end(), sort_index);
        // random shuffle the equivalent distance if more than nbest
        int nbest=(int)round(acc_rate*(double)nreps);
        if (all_dist[indices[0]]==all_dist[indices[nbest]]) {
            unsigned int max_index=nbest;
            while(all_dist[indices[0]]==all_dist[indices[max_index]] && max_index<nreps) {
                max_index++;
            }
            omptl::random_shuffle(indices.begin(), indices.begin()+max_index);
        }
        
        // write out the best simulations
        for (int i=0;i<nbest;++i) {
            s_out << res_s[indices[i]] << " ";
            if (min_h!=max_h || simulations_provided) h_out << res_h[indices[i]] << " ";
            if (N_boot) N_out << res_N[indices[i]] << " " ;
        }
        s_out << endl;
        if (min_h!=max_h || simulations_provided) h_out << endl;
        if (N_boot) N_out << endl;
        
        if (write_last_sims && l==(nloci-1)) {
            ofstream sims_retained_out((file_prefix + "_sim_retained_stats.txt").c_str());
            sims_retained_out << "Fsi" << "\t" << "Fsd" << "\t" << "Fsi_l" << "\t" << "Fsd_l" << "\t" << "tl" << "\t" << "th" << "\t" << "s" << "\t" << "h" << "\t"<< "Ne" << endl;
            for (int i=0;i<nbest;++i) {
                sims_retained_out << res_fsi[indices[i]] << "\t" << res_fsd[indices[i]] << "\t" << res_fsi_l[indices[i]] << "\t" << res_fsd_l[indices[i]] << "\t" << res_tl[indices[i]] << "\t" << res_th[indices[i]] << "\t" << res_s[indices[i]] << "\t" << res_h[indices[i]] << "\t" << res_N[indices[i]] << endl;
            }
            sims_retained_out.close();
        }
        
    } // end of loop over all loci
    
    // write output
    if (write_last_sims) {
        ofstream sims_out((file_prefix + "_sim_stats.txt").c_str());
        sims_out << "Fsi" << "\t" << "Fsd" << "\t" << "Fsi_l" << "\t" << "Fsd_l" << "\t" << "tl" << "\t" << "th" << "\t" << "s" << "\t" << "h" << "\t"<< "Ne" << endl;
        for (unsigned int rep = 0; rep < nreps; ++rep) {
            sims_out << res_fsi[rep] << "\t" << res_fsd[rep] << "\t" << res_fsi_l[rep] << "\t" << res_fsd_l[rep] << "\t" << res_tl[rep] << "\t" << res_th[rep] << "\t" << res_s[rep] << "\t" << res_h[rep] << "\t" << res_N[rep] << endl;
        }
        sims_out.close();
    }

    s_out.close();
    if (min_h!=max_h || simulations_provided) h_out.close();
    if (N_boot) N_out.close();
    //tempo.close();
    //malas.close();

    return 0;
}
