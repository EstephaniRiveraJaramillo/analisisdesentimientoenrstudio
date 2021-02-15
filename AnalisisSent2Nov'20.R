#Análisis de los datos
#Clasificar emociones

rm(list = ls())
#############explorando 3###########################################
library(tidyverse)
library(tidytext)
library(naivebayes)
library(tm)
library(caret)
library(dplyr)
library(MLmetrics)
library(corrplot)
library(mlr)
library(sqldf)
library(ggplot2)
library(haven)
library(pROC)
library(reshape2)
library(tidyr)
library(purrr)
library(widyr)
library(ggraph)
library(igraph)
library(topicmodels)
library(lubridate)
library(wordcloud)
library(readr)

#llaves twitter
#consumerKey = "glmxBK83K4VXUMlzPS3zlBMP3"
#consumerSecret = "0PS9XUDT2Tcg8wiy9dFGXwJYsyXDqk2GwiSQsFvkLwhnSIEsKq"
#accessToken = "1270690437884502016-j8QRfAjwruNWMJn0bfuv04zQhjdMEU"
#accessSecret = "1g6dAvQ1WkVxCM5OvbEBWeMgmncENlsGICTFAwUnNPOyj"
#options(httr_oauth_cache=TRUE)
#setup_twitter_oauth(consumer_key = consumerKey, consumer_secret = consumerSecret,
#                    access_token = accessToken, access_secret = accessSecret)

####################
#fn_twitter1 <- searchTwitter("@George_Forsyth",n=300,lang="es")
#fn_twitter3 <- searchTwitter("@CesarAcunaP",n=300,lang="es")
#fn_twitter4 <- searchTwitter("@julioguzmanperu",n=300,lang="es")

#convierte a dataframe
#e <- twListToDF(fn_twitter1)
#f <- twListToDF(fn_twitter3)
#g <- twListToDF(fn_twitter4)

#setwd("D:/Capacitaciones de Estephani Rivera/Analisis de Sentimientos para URP")
#exportar
#write.csv(e,"salida/e.csv")
#write.csv(f,"salida/f.csv")
#write.csv(g,"salida/g.csv")

#importar

setwd("D:/Capacitaciones de Estephani Rivera/Analisis de Sentimientos para URP/salida")

tweets1 <- read.csv("e.csv",sep =";")
tweets2 <- read.csv("f.csv",sep =";")
tweets3 <- read.csv("g.csv",sep =";")

# Se unen todos los tweets en un único dataframe
tweets <- bind_rows(tweets1,tweets2,tweets3)

library(srvyr)

tweets %>% group_by(screen_Name) %>% summarise(numero_tweets = n()) 

colnames(tweets)

tweets$ID=seq.int(nrow(tweets))
names(tweets)
tuits_df_descrip = tweets[,c(1,2,6,9,18)]
names(tuits_df_descrip)[3] <- "fecha"


###tabla por screen_name###
table(tuits_df_descrip$screen_Name)

####vista preliminar#####
head(tuits_df_descrip)

######limpieza y tokenizacion de las palabras##############
##seleccion de columnas
tweets <- tuits_df_descrip %>% select(screen_Name,id,text,fecha)

tweets <- tweets %>% rename(texto = text)
##limpieza##
limpiar_tokenizar <- function(texto){
  # El orden de la limpieza no es arbitrario
  # Se convierte todo el texto a minúsculas
  nuevo_texto <- tolower(texto)
  # Eliminación de páginas web (palabras que empiezan por "http." seguidas 
  # de cualquier cosa que no sea un espacio)
  nuevo_texto <- str_replace_all(nuevo_texto,"http\\S*", "")
  # Eliminación de signos de puntuación
  nuevo_texto <- str_replace_all(nuevo_texto,"[[:punct:]]", " ")
  # Eliminación de números
  nuevo_texto <- str_replace_all(nuevo_texto,"[[:digit:]]", " ")
  # Eliminación de espacios en blanco múltiples
  nuevo_texto <- str_replace_all(nuevo_texto,"[\\s]+", " ")
  # Eliminacion de tildes
  nuevo_texto <- chartr('áéíóúñ','aeioun',nuevo_texto)
  # Tokenización por palabras individuales
  nuevo_texto <- str_split(nuevo_texto, " ")[[1]]
  # Eliminación de tokens con una longitud < 2
  #nuevo_texto <- keep(.x = nuevo_texto, .p = function(x){str_length(x) > 1})
  return(nuevo_texto)
}

tweets <- tweets %>% mutate(texto_tokenizado = map(.x = texto,
                                                   .f = limpiar_tokenizar))
tweets %>% select(texto_tokenizado) %>% head()

###vista previa del tuits 100##
tweets %>% slice(100) %>% select(texto_tokenizado) %>% pull()
###transformacion del archivo##
tweets_tidy <- tweets %>% select(-texto) %>% unnest()
tweets_tidy <- tweets_tidy %>% rename(token = texto_tokenizado)
###se creo la matriz donde cada palabra de cada tuit sera una fila
# Se filtran las stopwords
data(stop_words)
tweets_tidy <- tweets_tidy %>% filter(!(token %in% stop_words$word))
head(tweets_tidy) 

#Una forma de analizar el sentimiento de un de un texto es considerando su sentimiento como el la suma de los sentimientos de cada una de las palabras que lo forman. 
#Esta no es la única forma abordar el análisis de sentimientos, pero consigue un buen equilibro entre complejidad y resultados.
#Para llevar a cabo esta aproximación es necesario disponer de un diccionario en el que se asocie a cada palabra un sentimiento o nivel de sentimiento. 
#A estos diccionarios también se les conoce como sentiment lexicon. El paquete tidytext contiene 3 diccionarios distintos:

#AFINN: asigna a cada palabra un valor entre -5 y 5. Siendo -5 el máximo de negatividad y +5 el máximo de positividad.
#bing: clasifica las palabras de forma binaria positivo/negativo.
#nrc: clasifica cada palabra en uno o más de los siguientes sentimientos: positive, negative, anger, anticipation, disgust, fear, joy, sadness, surprise, and trust.
#En este ejercicio se emplea la clasificación positivo/negativo proporcionada por el diccionario bing.

sentimientos <- get_sentiments(lexicon = "bing")
head(sentimientos)

sentimientos <- sentimientos %>%
  mutate(valor = if_else(sentiment == "negative", -1, 1))

tweets_sent <- inner_join(x = tweets_tidy, y = sentimientos,
                          by = c("token" = "word"))

#Sentimiento promedio de cada tweet
tweets_sent %>% group_by(screen_Name,id) %>%
  summarise(sentimiento_promedio = sum(valor)) %>%
  head()


#Porcentaje de tweets positivos, negativos y neutros por autor
tweets_sent %>% group_by(screen_Name,id) %>%
  summarise(sentimiento_promedio = sum(valor)) %>%
  group_by(screen_Name) %>%
  summarise(positivos = 100 * sum(sentimiento_promedio > 0) / n(),
            neutros = 100 * sum(sentimiento_promedio == 0) / n(),
            negativos = 100 * sum(sentimiento_promedio  < 0) / n())


a=tweets_sent %>% group_by(screen_Name,id) %>%
  summarise(sentimiento_promedio = sum(valor)) %>%
  group_by(screen_Name) %>%
  summarise(positivos = 100*sum(sentimiento_promedio > 0) / n(),
            neutros = 100*sum(sentimiento_promedio == 0) / n(),
            negativos = 100*sum(sentimiento_promedio  < 0) / n()) %>%
  ungroup() %>%
  gather(key = "sentimiento", value = "valor", -screen_Name) %>%
  ggplot(aes(x = screen_Name, y = valor, fill = sentimiento)) + 
  geom_col(position = "dodge", color = "black") + coord_flip() +
  theme_bw()

library(plotly)
ggplotly(a)








