Wrappers for tools that estimate selection coefficient using time series genomic data
================
Erik Zhivkoplias
September 10, 2020






## Introduction

This repo provides a set of scripts to ensure reproducibility of the results available here
(<https://www.diva-portal.org/smash/get/diva2:1469953/FULLTEXT01.pdf>).
In this study we aimed to compare the performance of different methods
to estimate selection coefficient across parameter space.

We to compared the performance of four tools:

 - WFABC 1.1. (<http://jjensenlab.org/wp-content/uploads/2016/02/WFABC_v1.1.zip>)
 - CLEAR (<https://github.com/airanmehr/clear>)
 - LLS (<https://github.com/ThomasTaus/poolSeq/>)
 - slattice (<https://github.com/mathii/slattice>)
 
Other methods/tools could be added to the comparisson set later.


## Computational Setup

The following Unix tools and R packages need to be set up as follows:

### UNIX environment

Before running any of the pipelines below, you will first need to ensure
that your UNIX environment is properly configured. It is required to install
Anaconda management system (<https://docs.conda.io/en/latest/>, and create a 
Python2-based virtual environment:

``` bash
conda create -n py2 --file py2-package-list.txt
```

Make sure to keep the correct name (py2). You don't have to activate the environment
before starting the R session.


### R packages

Make sure to clone github repos and install LLS and slattice R packages according to instroctions provided.
After that, start an R session and install the following packages:

``` r
install.packages(c("bit64","poolSeq", "Metrics", "ggplot2", "reshape2", "reticulate", "plyr",
 "dplyr", "slattice", "Biobase", "functional", "hash", "data.table", "HistogramTools",
  "tidyr", "plotly", "corrplot", "philentropy", "sfsmisc", "akima"))
```

### Tools

We also provide the relevant versions of WFABC 1.1. and CLEAR compared in present study.
Given that the first one is not currently present on github, it might be tricky to find
a relevant version in future.

As for CLEAR, the code has been slightly modified for the sake of our goals. In particular, the range
of values in the precomputeTransition function (CLEAR.py) was increased, while decreasing the
for transition step. It was done to ensure the comparability with other tools that use the
continious space of selection coefficient estimates.

Good luck!

