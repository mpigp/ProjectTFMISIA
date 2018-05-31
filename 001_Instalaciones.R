#Instalaciones
install.packages("devtools")
devtools::install_github("rstudio/keras")
library(keras)
install_keras()
install_tensorflow()
install.packages("rminer")
install.packages("reticulate")
reticulate::py_config()


pkgs <- c("lime", "tidyquant", "rsample", "recipes", "yardstick", "corrr")
install.packages(pkgs)

require(devtools)
install_github("rstudio/reticulate")
devtools::install_github("rstudio/tensorflow")
install_github("rstudio/keras")
