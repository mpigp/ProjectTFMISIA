#Comprobamos que estamos en el directorio correcto
# Obtenemos la información de donde estamos situados
DirectorioActual <- getwd()
DirectorioNecesario <- "C:/Users/PILAR/Documents/ProjectTFMISIA"

# En el caso que estemos donde queremos lo dejamos tal cual, si no estamos en el directorio correcto nos cambiamos
if(DirectorioActual == DirectorioNecesario) {
  (print('Estamos en el directorio correcto'))
}else{
  print('Cambio de directorio')
  setwd("C:")
  workingDir <- "C:/Users/PILAR/Documents/ProjectTFMISIA"
  setwd(workingDir)
}