
ACC <- function(v1,v2){
  #' @title Realiza el calculo del Accuary
  #' @description Realiza el calculo del Accuary dados dos vectores
  #' @param v1 vector que contiene lo obtenido
  #' @param v2 vector con el que se compara
  #' @return ACC
  #' @author Pilar Guerrero
  
  ok <- 0
  for(k in 1:length(v1)){
    if (v1[k]==v2[k]){
      ok = ok +1
    }
  }
  return(ok/length(v1))
}