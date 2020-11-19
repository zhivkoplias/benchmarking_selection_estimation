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

#include "omptl/omptl_algorithm"

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

using namespace std;

// http://www.concentric.net/~Ttwang/tech/inthash.htm
unsigned long mix(unsigned long a, unsigned long b, unsigned long c) {
    a = a - b; a = a - c; a = a^(c >> 13); b = b - c; b = b - a; b = b^(a << 8); c = c - a; c = c - b; c = c^(b >> 13); a = a - b; a = a - c; a = a^(c >> 12); b = b - c; b = b - a; b = b^(a << 16); c = c - a; c = c - b; c = c^(b >> 5); a = a - b; a = a - c; a = a^(c >> 3); b = b - c; b = b - a; b = b^(a << 10); c = c - a; c = c - b; c = c^(b >> 15); return c;
}

int main(int argc, char **argv) {

    unsigned int seed = 0;
    unsigned int nboots = 10000; // number of bootstrap replicates
    double iqr = 3;
    double min_freq_fs = 0.01; // minimum frequency to calculate Fs'
    double freq_threshold = 0.05; // used to calculate time below this freq (used to estimate h)
    double fs_threshold = 0.00; // Minor frequency threshold to decompose Fsi and Fsd


    string file_name;

    AnyOption *opt = new AnyOption();

    opt->noPOSIX();
    //opt->setVerbose();
    //opt->autoUsagePrint(true);

    opt->setFlag("help",'h');
    opt->addUsage( " ----------------------------- " );
	opt->addUsage( " | WFABC_1 1.0 usage:        | " );
    opt->addUsage( " |  values indicate defaults | " );
    opt->addUsage( " |  n indicates integer      | " );
	opt->addUsage( " ----------------------------- " );
    opt->addUsage( " input.txt         Name of the input file " );
    opt->addUsage( " -help             Prints this help " );
    opt->addUsage( " -nboots 10000     Number of bootstrap replicates " );
    opt->addUsage( " -nthreads n       Number of threads (automatically detected by default) " );
    opt->addUsage( " ---------------------------------------------- " );
	opt->addUsage( " | Advanced options (change with caution)     | " );
	opt->addUsage( " ---------------------------------------------- " );
    opt->addUsage( " -min_freq_fs 0.01 Minimum allele frequency to calculate the Fs statistics " );
    opt->addUsage( " -F_threshold 0.00 Minor frequency threshold to decompose Fsi and Fsd" );
    opt->addUsage( " -t_threshold 0.05 Minor frequency threshold to calculate tl and th" );
    opt->addUsage( " -iqr 3            Inter-Quartile Region to consider for a robust estimate of Ne " );
    opt->addUsage( " -seed n           Manually set the seed (used for debbuging)" );
    
    opt->setOption("seed");
    opt->setOption("nboots");
    opt->setOption("iqr");
    opt->setOption("min_freq_fs");
    opt->setOption("F_threshold");
    opt->setOption("t_threshold");
    opt->setOption("nthreads");

    opt->processCommandArgs(argc, argv);

    if (opt->getArgc() == 0) { /* print usage if no options */
        opt->printUsage();
        delete opt;
        return 1;
    }
    if (opt->getFlag("help"))
        opt->printUsage();

    if (opt->getValue("seed") != NULL)
        seed = atoi(opt->getValue("seed"));
    if (opt->getValue("nboots") != NULL)
        nboots = atoi(opt->getValue("nboots"));
    if (opt->getValue("iqr") != NULL)
        iqr = atof(opt->getValue("iqr"));
    if (opt->getValue("min_freq_fs") != NULL)
        min_freq_fs = atof(opt->getValue("min_freq_fs"));
    if (opt->getValue("t_threshold") != NULL)
        freq_threshold = atof(opt->getValue("t_threshold"));
    if (opt->getValue("F_threshold") != NULL)
        fs_threshold = atof(opt->getValue("F_threshold"));
    
    file_name = opt->getArgv(0);

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
    
    unsigned int nb_times; // numer of time points
    unsigned int nloci; // number of loci
    // read intput file
    ifstream infile(file_name.c_str());
    string line;
    if (infile.is_open())
    {
        string file_prefix = file_name;
        if (file_prefix.find(".", 1) != string::npos)
            file_prefix = file_prefix.substr(0, file_prefix.length() - 4);
//        unsigned long position;
//        if ((position = file_prefix.find_last_of('/')))
//            file_prefix = file_prefix.substr(position + 1, file_prefix.length());

        ofstream stats_out((file_prefix + "_obs_stats.txt").c_str());

        stats_out << "Fsi" << "\t" <<  "Fsd" << "\t" << "Fsi_l" << "\t" <<  "Fsd_l" << "\t" << "tl" << "\t" << "th" << endl;
        
        // read number of loci and number of times
        getline (infile,line);
        istringstream header_line(line);
        header_line >> nloci;
        header_line >> nb_times;
        // read time points
        getline (infile,line);
        istringstream time_line(line);
        vector<int> sample_times(nb_times); // time of the samples (in generations)
        
        unsigned int** N_sample = new unsigned int*[nloci]; // sample sizes at each time point (nb of individuals)
        for (unsigned int i = 0; i < nloci; ++i)
            N_sample[i] = new unsigned int[nb_times];
        
        //vector<unsigned int> N_sample(nb_times); // sample sizes at each time point (nb of individuals)
        vector<unsigned int> sample_count(nb_times);
        vector<double> sample_freq(nb_times);
        int cur_time;
        for (unsigned int i = 0; i < nb_times; ++i) {
            time_line >> cur_time;
            sample_times[i]=cur_time;
            time_line.ignore(1);
        }
        vector<double> fsp(nb_times - 1);
        vector<double> res_fsd(nloci);
        vector<double> res_fsi(nloci);
        vector<double> res_fsd_l(nloci);
        vector<double> res_fsi_l(nloci);
        vector<double> res_tl(nloci);
        vector<double> res_th(nloci);
        double** res_fs_num = new double*[nb_times - 1];
        for (unsigned int i = 0; i < nb_times - 1; ++i)
            res_fs_num[i] = new double[nloci];
        double** res_fs_den = new double*[nb_times - 1];
        for (unsigned int i = 0; i < nb_times - 1; ++i)
            res_fs_den[i] = new double[nloci];
        double x_fs, y_fs, z_fs, fsi, fsd, fsi_l, fsd_l, tl, th, nt_fs;
        unsigned int ngen_fs,ngen_fs_l;
        // read each locus one by one
        for (unsigned int l = 0; l < nloci; ++l) {
            getline (infile,line);
            istringstream cur_line_N_sample(line);
            // read sample sizes
            int cur_N_sample;
            for (unsigned int i = 0; i < nb_times; ++i) {
                cur_line_N_sample >> cur_N_sample;
                N_sample[l][i]=cur_N_sample;
                cur_line_N_sample.ignore(1);
            }
            // read number of alleles
            getline (infile,line);
            istringstream cur_line_sample(line);
            int cur_sample;
            for (unsigned int i = 0; i < nb_times; ++i) {
                cur_line_sample >> cur_sample;
                sample_count[i]=cur_sample;
                cur_line_sample.ignore(1);
                sample_freq[i]=(double) cur_sample / (double) (N_sample[l][i]);
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
                // check the mini frequency condition and calculate Fs' if possible
                if (!(((x_fs < min_freq_fs) & (y_fs < min_freq_fs)) || (((1 - x_fs) < min_freq_fs) & ((1 - y_fs) < min_freq_fs)))) {
                    z_fs = (x_fs + y_fs) / 2;
                    res_fs_num[i - 1][l] = (x_fs - y_fs);
                    res_fs_den[i - 1][l] = (z_fs);
                    fsp[i - 1] = res_fs_num[i - 1][l] * res_fs_num[i - 1][l] / (res_fs_den[i - 1][l] * (1 - res_fs_den[i - 1][l]));
                    nt_fs = 1 / ((1 / (double) (N_sample[l][i - 1]) + 1 / (double) (N_sample[l][i])) / 2);
                    fsp[i - 1] = ((fsp[i - 1]*(1 - 1 / (2 * nt_fs)) - 2 / nt_fs) / ((1 + fsp[i - 1] / 4)*(1 - 1 / ((double) (N_sample[l][i])))));
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
                } else { // indicates that this locus and time is excluded
                    res_fs_num[i - 1][l] = -1;
                    res_fs_den[i - 1][l] = -1;
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

            res_fsi[l] = fsi;
            res_fsd[l] = fsd;
            res_fsi_l[l] = fsi_l;
            res_fsd_l[l] = fsd_l;
            res_tl[l] = tl;
            res_th[l] = th;
            stats_out << res_fsi[l] << "\t" <<  res_fsd[l] << "\t" << res_fsi_l[l] << "\t" <<  res_fsd_l[l] << "\t" <<  res_tl[l] << "\t" <<  res_th[l] << endl;

        }
        // bootstrap for Ne
        if (nboots > 0) {
            ofstream Ne_out;
            if (nboots > 0) {
                Ne_out.open((file_prefix + "_Ne_bootstrap.txt").c_str());
            }
            double boot_mean_fsp = 0;
            double boot_sd_fsp = 0;
            vector<double> boot_Ne(nboots);
            
            // for each pair of time points, find the cutoff values for the trimmed mean of the numerator and denominator of Fs
            vector<double> lb_num_trim(nb_times - 1);
            vector<double> hb_num_trim(nb_times - 1);
            vector<double> lb_den_trim(nb_times - 1);
            vector<double> hb_den_trim(nb_times - 1);
#pragma omp parallel for
            for (unsigned int i = 1; i < nb_times; ++i) {
                // create a vector for num and den and exclude the -1
                vector<double> sort_num;
                vector<double> sort_den;
                for (unsigned int rep = 0; rep < nloci; ++rep) {
                    if (res_fs_num[i - 1][rep] != -1) {
                        sort_num.push_back(res_fs_num[i - 1][rep]);
                        sort_den.push_back(res_fs_den[i - 1][rep]);
                    }
                }
                // sort the vectors
                omptl::sort(sort_num.begin(), sort_num.end());
                omptl::sort(sort_den.begin(), sort_den.end());
                // get the quantiles
                double quart1, quart3;
                if (iqr > 0 && sort_num.size()>0) {
                    quart1 = sort_num[round(0.25 * (double) (sort_num.size()))];
                    quart3 = sort_num[round(0.75 * (double) (sort_num.size())) - 1];
                    lb_num_trim[i - 1] = quart1 - iqr * (quart3 - quart1);
                    hb_num_trim[i - 1] = quart3 + iqr * (quart3 - quart1);
                } else { // zero or negative iqr means no trim
                    lb_num_trim[i - 1] = -1;
                    hb_num_trim[i - 1] = 1;
                }
                if (iqr > 0  && sort_den.size()>0) {
                    quart1 = sort_den[round(0.25 * (double) (sort_den.size()))];
                    quart3 = sort_den[round(0.75 * (double) (sort_den.size())) - 1];
                    lb_den_trim[i - 1] = quart1 - iqr * (quart3 - quart1);
                    hb_den_trim[i - 1] = quart3 + iqr * (quart3 - quart1);
                } else { // zero or negative iqr means no trim
                    lb_den_trim[i - 1] = 0;
                    hb_den_trim[i - 1] = 1;
                }
            }
         
#pragma omp parallel
            {
                // Make the average over loci for each time point to calculate Fs'
                double sum_num, sum_den;
                vector<double> mean_fsp(nb_times - 1);
                double all_mean_fsp;
                double nt_fs;
                double harm_x,harm_y;
                int num_harm;
                //int nzero;
                // generate random weights from dirichlet(1,,,1) to do a bayesian bootstrap
                // dirichlet can be generated using gamma (http://en.wikipedia.org/wiki/Dirichlet_distribution#Gamma_distribution) and dividing by the sum
                // for a dirichlet(1,,,1), one need to generate from gamma(1,1) which is actually an exp(1) (http://en.wikipedia.org/wiki/Gamma_distribution#Generating_gamma-distributed_random_variables)
                // exp(1) can be generated using -ln(uniform(0,1))) (http://en.wikipedia.org/wiki/Exponential_distribution#Generating_exponential_variates)
                vector<double> w_num(nloci);
                vector<double> w_den(nloci);
                double sum_w_num, sum_w_den;
#pragma omp for reduction(+:boot_mean_fsp,boot_sd_fsp)
                for (unsigned int boot = 0; boot < nboots; ++boot) {
                    all_mean_fsp = 0;
                    for (unsigned int i = 1; i < nb_times; ++i) {
                        sum_num = 0;
                        sum_den = 0;
                        sum_w_num = 0;
                        sum_w_den = 0;
                        harm_x = 0;
                        harm_y = 0;
                        num_harm=0;
                        for (unsigned int rep = 0; rep < nloci; ++rep) {
                            if (res_fs_num[i - 1][rep] != -1 && res_fs_num[i - 1][rep] >= lb_num_trim[i - 1] && res_fs_num[i - 1][rep] <= hb_num_trim[i - 1])
                                w_num[rep] = -log(runif(0, 1));
                            else w_num[rep] = 0;
                            sum_w_num += w_num[rep];
                        }
                        for (unsigned int rep = 0; rep < nloci; ++rep) {
                            if (res_fs_den[i - 1][rep] != -1 && res_fs_den[i - 1][rep] >= lb_den_trim[i - 1] && res_fs_den[i - 1][rep] <= hb_den_trim[i - 1])
                                w_den[rep] = -log(runif(0, 1));
                            else w_den[rep] = 0;
                            sum_w_den += w_den[rep];
                        }
                        for (unsigned int rep = 0; rep < nloci; ++rep) {
                            w_num[rep] /= sum_w_num;
                            w_den[rep] /= sum_w_den;
                            
                        }
                        for (unsigned int rep = 0; rep < nloci; ++rep) {
                            sum_num += w_num[rep] * res_fs_num[i - 1][rep] * res_fs_num[i - 1][rep];
                            sum_den += w_den[rep] * res_fs_den[i - 1][rep] * (1 - res_fs_den[i - 1][rep]);
                        }
                        // calculate harmonic mean of sample sizes
                        for (unsigned int rep = 0; rep < nloci; ++rep) {
                            if (res_fs_num[i - 1][rep] != -1) {
                                harm_x+=1/(double)N_sample[rep][i - 1];
                                harm_y+=1/(double)N_sample[rep][i];
                                num_harm++;
                            }
                        }
                        harm_x=num_harm/harm_x;
                        harm_y=num_harm/harm_y;
                        nt_fs=1 / ((1 / harm_x + 1 / harm_y) / 2);
                        mean_fsp[i-1] = sum_num / sum_den;
                        //nt_fs = 1 / ((1 / (double) (N_sample[l][i - 1]) + 1 / (double) (N_sample[l][i])) / 2);
                        mean_fsp[i-1] = ((mean_fsp[i-1]*(1 - 1 / (2 * nt_fs)) - 2 / nt_fs) / ((1 + mean_fsp[i-1] / 4)*(1 - 1 / harm_y)));
                        //cout << mean_fsp[i-1] << endl;
                        if (!isnan(mean_fsp[i-1])) {
                            all_mean_fsp += mean_fsp[i-1];
                        }
                    }
                    all_mean_fsp /= sample_times[nb_times - 1] - sample_times[0];
                    boot_Ne[boot]=1/(all_mean_fsp);
                    boot_mean_fsp += all_mean_fsp;
                    boot_sd_fsp += (all_mean_fsp)*(all_mean_fsp);
                }
            }
            boot_mean_fsp /= nboots;
            boot_sd_fsp = sqrt(boot_sd_fsp / nboots - boot_mean_fsp * boot_mean_fsp);
            cout << 1/boot_mean_fsp << "\t" << boot_sd_fsp/(boot_mean_fsp*boot_mean_fsp) << endl;
            for (unsigned int boot = 0; boot < nboots; ++boot) {
               Ne_out << boot_Ne[boot] << endl;
            }
            Ne_out.close();
        
        }
        
        for (int unsigned i = 0; i < nb_times - 1; ++i) {
            delete [] res_fs_num[i];
        }
        delete [] res_fs_num;
        for (int unsigned i = 0; i < nb_times - 1; ++i) {
            delete [] res_fs_den[i];
        }
        delete [] res_fs_den;
        infile.close();
        stats_out.close();
    }
    
    else cout << "Unable to open file: " << file_name << endl;

    return 0;
}
