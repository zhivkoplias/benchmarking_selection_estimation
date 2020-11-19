# launch WFABC step 1 on the simulated data
# run this in the terminal or directly in R (not working in Windows)
system("./wfabc_1 multiple_loci.txt ")
# plot the posterior for Ne
post_N=read.table("multiple_loci_Ne_bootstrap.txt")
plot(density(post_N[,1]),lwd=2,main="Ne Posterior",xlab="Ne")
# launch WFABC step 2 on the simulated data
# run this in the terminal or directly in R (not working in Windows)
system("./wfabc_2 -min_s -0.5 -max_s 0.5 multiple_loci.txt")
# plot the posterior for s at locus 1 and 1000
post_s=read.table("multiple_loci_posterior_s.txt")
plot(density(t(post_s[1,])),lwd=2,main="s Posterior (locus 1)",xlab="s")
plot(density(t(post_s[1000,])),lwd=2,main="s Posterior (locus 1000)",xlab="s")

# To plot a 2D posterior for s and Ne
post_N=read.table("multiple_loci_posterior_N.txt")
require(MASS)
z <- kde2d(t(post_s[1000,]),t(post_N[1000,]),n=300)
image(z,xlab="s",ylab="Ne",main="2D posterior (locus 1000)") 
# to find the mode in the 2D posterior
maxindex=which(z$z==max(z$z), arr.ind=T) 
z$x[maxindex[1]]
z$y[maxindex[2]]

# calculate the posterior mean for each locus
rowMeans(post_s)
# obtain 95% confidence intervals (Highest Posterior Density Intervals)
require(boa)
apply(post_s,1,boa.hpd,1-0.95)

# make two boxplot for the first 990 loci (s=0) and the last 10 (s=0.1)
boxplot(rowMeans(post_s[1:990,]),rowMeans(post_s[991:1000,]),names=c("neutral","selected"))



#############################################################################
# This section shows how the dataset "multiple_loci.txt" has been simulated
#############################################################################
# load the WF simulation function
source("WF_simulation.r")
# define s: 990 neutral loci (s=0) and 10 selected (s=0.1)
s=c(rep(0,l=990),rep(0.1,l=10))
# output file name (will be the input of WFABC)
file_out="multiple_loci.txt"
# write the header (nb of loci and nb of time points)
cat(length(s),12,"\n",file=file_out)
for (k in 1:length(s)) {
  # simulate one trajectory
  res=WF_trajectory(N=1000,t=100,j=500,s=s[k],N_sample=100,nb_times=12,ploidy=2)  
  # write the times only for the first one
  if (k==1) cat(res[["times"]],"\n",file=file_out,append=T,sep=",")
  # write sample size
  cat(res[["N_sample"]],"\n",file=file_out,append=T,sep=",") 
  # write number of alleles
  cat(res[["N_A2"]],"\n",file=file_out,append=T,sep=",")
}


