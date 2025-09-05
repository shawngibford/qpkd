# R package requirements for quantum PK/PD modeling
# Install these packages in R before using the Python interface

# Core nlmixr2 ecosystem
install.packages("nlmixr2")
install.packages("nlmixr2extra")  
install.packages("nlmixr2plot")
install.packages("xpose.nlmixr2")
install.packages("ggPMX")

# Core R packages for PK/PD
install.packages("dplyr")
install.packages("ggplot2")
install.packages("tidyr")
install.packages("readr")

# Statistical modeling
install.packages("lme4")
install.packages("nlme")
install.packages("MASS")

# Pharmacometrics specific
install.packages("vpc")  # Visual predictive checks
install.packages("mrgsolve")  # ODE-based modeling
install.packages("PKPDmodels")
install.packages("RxODE")

# Data manipulation and visualization
install.packages("data.table")
install.packages("lattice")
install.packages("gridExtra")

# Development version installation (if needed):
# remotes::install_github("nlmixr2/nlmixr2")