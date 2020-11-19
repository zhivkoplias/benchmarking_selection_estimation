# simulates the trajectory of a Wright-Fisher model with selection
# N: population size (number of chromosomes: N individuals for haploids and N/2 individuals for diploids)
#    N can be set as a function, in which case a new value is randomly drawn from it each time a trajectory is simulated,
#    including when it's relaunched because the min_freq condition is not met (see below)
# t: number of generations
# t0: time where the mutation appears in generations (1 by default, appears at the begining)
# j: number of A alleles appearing at time t0 in the population (1 for a de novo mutation by default)
# s, h: selection coefficient and dominance (wA=1+s;wa=1 for haploids and wAA=1+s;wAa=1+s*h;waa=1 for diploids)
#       s can be set as a function, in which case a new value is randomly drawn from it each time a trajectory is simulated,
#       including when it's relaunched because the min_freq condition is not met (see below)
# s_start: time in generation when selection starts (s=0 before that).
# ploidy: 1 for haploids, 2 for diploids 
# min_freq: condition on the mutation reaching a certain frequency (put 0 for no condition and 1 to condition on fixation), 
#           runs several simulations until the criterion is met, so can take a longer time
# time sampling: 
# nb_times: number of equaly spaced time points to use between 1 ant t
# N_sample: sample size (in number of chromosomes). Can be a number or a vector of size nb_times
# sample_times: gives the exact sampling times in generations (not compatible with nb_times)
# max_sims: maximum number of simulations to do before giving up (usefull when condition on polymorphism or min_freq is unrealistic)
WF_trajectory<-function(N,t,j=1,t0=1,s=0,h=0.5,s_start=1,ploidy=1,min_freq=0,poly_condition=FALSE,nb_times=0,N_sample=100,sample_times=c(),max_sims=10000) { 
  # check if we will have to do time sampling
  time_sampling=(nb_times>0 | length(sample_times)>0)
  if (nb_times>0 & length(sample_times)>0) {
    stop("Please provide either nb_times or sample_times but not both")
  }
  if(time_sampling) {
    if(length(sample_times)==0) { 
      sample_times=round(seq(1,t,length=nb_times))
    } else {
      nb_times=length(sample_times)
    }
  }
  if (length(N_sample)==1) N_sample=rep(N_sample,length=nb_times)
  nsims=0
  repeat{
    nsims=nsims+1
    # initialize the trajectory
    x=numeric(t) 
    x[t0]=j
    # set selection coefficient
    if(is.function(s)) cur_s=s() else cur_s=s    
    # set Ne
    if(is.function(N)) cur_N=N() else cur_N=N    
    # loop over generations
    for (i in (t0+1):t)
    {
      # set s
      my_s=cur_s
      if(i<=s_start) my_s=0
      # calculate fitness
      wAA=1+my_s;wAa=1+my_s*h;waa=1
      wA=1+my_s;wa=1
      p=x[i-1]/(cur_N)
      # calculate sampling probbilities
      if (ploidy==2) {  
        prob=(wAA*p^2+wAa*p*(1-p))/(wAA*p^2+wAa*2*p*(1-p)+waa*(1-p)^2)
      }
      else {
        prob=wA*p/(wA*p+wa*(1-p))      
      }    
      # binomial sampling to the next generation
      x[i]=rbinom(1,cur_N,prob)
      # stop if allele is lost
      if (x[i]==0) break
    }
    # do time sampling
    if(time_sampling) {
      sample_freq=rbinom(nb_times,N_sample,x[sample_times]/(cur_N))
    }
    # allele frequency and polymormic condition
    if (time_sampling) {
      if ((sample_freq[nb_times]/(N_sample[nb_times]))>=min_freq){ 
        if (poly_condition) {
          if(sample_freq[nb_times]!=0 & sample_freq[nb_times]!=N_sample[nb_times]) {
            break
          }
        }
        else {
          break
        }
      }
    } else {
      if((x[t])>=min_freq*cur_N){
        break
      }
    }
    if (nsims>=max_sims) break
  }
  
  # create result list
  res=list("s"=cur_s,"N"=cur_N)
  if (time_sampling) {
    res[["N_A2"]]=sample_freq
    res[["N_sample"]]=N_sample
    res[["times"]]=sample_times    
  } else {
    res[["p"]]=x
  }
  res[["nsims"]]=nsims
  res
}
