#Instalaciones
install.packages("devtools")
devtools::install_github("rstudio/keras")
library(keras)
install_keras()
install_tensorflow()
install.packages("rminer")
install.packages("reticulate")
reticulate::py_config()

require(devtools)
install_github("rstudio/reticulate")
install_github("rstudio/tensorflow")
install_github("rstudio/keras")
