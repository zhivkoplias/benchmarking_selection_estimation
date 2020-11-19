# run this in the terminal or directly in R (not working in Windows)
system("./wfabc_1 -nboots 0 data_medionigra.txt")
system("./wfabc_2 -fixed_N 1000 -min_s -1 -max_s 0.1 data_medionigra.txt")

post_s=read.table("data_medionigra_posterior_s.txt")
plot(density(t(post_s[1,])),lwd=2,main="s Posterior",xlab="s")

# run this in the terminal or directly in R (not working in Windows)
system("./wfabc_2 -fixed_N 1000 -min_s -1 -max_s 0.1 -min_h 0.0 -max_h 1 data_medionigra.txt")
post_s=read.table("data_medionigra_posterior_s.txt")
post_h=read.table("data_medionigra_posterior_h.txt")
require(MASS);z <- kde2d(t(post_s[1,]),t(post_h[1,]),n=300)
image(z,xlab="s",ylab="h") 
