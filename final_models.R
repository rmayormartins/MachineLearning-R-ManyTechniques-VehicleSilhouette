################################# Libraries #################################
library(caret)
library(printr)
library("caTools")
library(rpart)
library(rpart.plot)
library(randomForest)
library(kernlab)
library(class)
library(e1071)
library(C50)
library(tidyverse)  
library(cluster)  
library(factoextra) 
library(NbClust)
library(vegan)

################################# Functions #################################
# K-means is a unsupervised method, so it isn't designed to predict on future data. But once we are using
# a training dataset with known classes, this functional approach suggested by Edward Visel was used.
# It looks for the closest center to the new point 
# params: obj is a model
#         newdata is the test data
predict.kmeans <- function(obj, newdata){
  centers <- obj$centers # Centros dos clustes
  n_centers <- nrow(centers) # N??mero de centros
  dist_mat <- as.matrix(dist(rbind(centers, newdata)))
  dist_mat <- dist_mat[-seq(n_centers), seq(n_centers)]
  max.col(-dist_mat)
}
#############################################################################
# params: obj is a model
define_clusters <- function(obj){
  classes=0
  for (i in 1:nrow(obj$centers)){
    model = 0
    cat=0
    for(j in 1:4){
      if (kmeans_tab[i,j] > model){
        cat = j
        model = kmeans_tab[i,j]
      }
    }
    classes[i]=cat
  }
  return(classes)
}
#############################################################################
# params: class is a vehicle type
#         table_t is a table. In this case, a confusion matrix
return_metrics <- function(table_t,class){
  s <-0
  e <-0
  if (class == "bus"){
    s <- table_t[1,1]/sum(table_t[,1])
    e <- sum(table_t[-1,-1])/(sum(table_t[-1,-1])+sum(table_t[1,-1]))
  }
  if (class == "opel"){
    s <- table_t[2,2]/sum(table_t[,2])
    e <- sum(table_t[-2,-2])/(sum(table_t[-2,-2])+sum(table_t[2,-2]))
  }
  if (class == "saab"){
    s <- table_t[3,3]/sum(table_t[,3])
    e <- sum(table_t[-3,-3])/(sum(table_t[-3,-3])+sum(table_t[3,-3]))
  }
  if (class == "van"){
    s <-table_t[4,4]/sum(table_t[,4])
    e <- sum(table_t[-4,-4])/(sum(table_t[-4,-4])+sum(table_t[4,-4]))
  }
  
  acuracy_b <- (s+e)/2
  
  return(c(s,e,acuracy_b))
} 
################################## ATTRIBUTES ################################# 


#[V1]	Compactness                    #[V2]	Circularity                        #[V3]	Distance Circularity
#[V4]	Radius Ratio                   #[V5]	Pr.axis Aspect Ratio               #[V6]	Max.length aspect ratio
#[V7]	Scatter Ratio                  #[V8]	Elong	elongatedness                #[V9]	Pr.axis rectangularity
#[V10]	Max.Length Rectangularity    #[V11] Scaled Variance Along Major Axis   #[V12]	Scaled variance along minor axis 
#[V13]	Scaled radius of gyration    #[V14]	Skewness about major axis          #[V15]	Skewness about minor axis 
#[V16]	Kurtosis about minor axis    #[V16]	Kurtosis about major axis          #[V18]	Hollows ratio
#[V19]	Class	type

############################## DATASET ############################# 

seed <- 123
metric <- "Accuracy"

# Le toda a tabela de veiculos
tabela <- read.csv('Vehicle.csv')
tabela <- na.omit(tabela)
tabelaClass <- tabela[19]

summary(tabela)
# Apresenta quantidade respectiva as classes de veiculos presentes no conjunto de dados
table(tabela$Class)

## Preparacao dos dados: Separa 80% para treinamento e 20% para teste
set.seed(seed)
grupos <- sample(2, nrow(tabela), replace=TRUE, prob=c(0.8, 0.2))
dados_treinamento <- tabela[grupos==1,]
dados_teste <- tabela[grupos==2,]

# Remove a coluna que identifica a classe de cada veiculo em ambos os grupos
dados_treinamento_sem_label <- dados_treinamento[-19]
dados_treinamento_sem_label_scaled = as.data.frame(lapply(dados_treinamento_sem_label, scale))
dados_teste_sem_label <- dados_teste[-19]
dados_teste_sem_label_scaled = as.data.frame(lapply(dados_teste_sem_label, scale))

# Guarda a classe que identifica cada veiculo em ambos os grupos
labels_dados_treinamento <- dados_treinamento$Class 
labels_dados_teste <- dados_teste$Class

####### Visualizando os atributos do dataset #######
# Plot de densidade
featurePlot(x = dados_treinamento_sem_label_scaled, 
            y = labels_dados_treinamento, 
            plot = "density",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")),
            auto.key=list(columns=3),
            pch = "|",
            layout = c(3, 6), 
            adjust = 1.5
)

############################## PARAMETROS FUNCAO TRAIN ############################## 

# Define funcao trainControl para validacao cruzada
#10-fold repetido 3 vezes
control <- trainControl(method = "repeatedcv", number = 10, repeats = 3, savePredictions = T, classProbs = T, returnResamp = "all")
############################## NAIVE BAYES #########################
set.seed(seed)
model_nbayes1 <- train(Class~., data=dados_treinamento, trControl=control, method="nb", preProcess = c("center","scale"),metric=metric)
set.seed(seed)
model_nbayes2 <- train(Class~., data=dados_treinamento, trControl=control, method="naive_bayes", preProcess = c("center","scale"),metric=metric)
############################## SVM ############################## 
set.seed(seed)
model_svm1 <- train(Class~., data=dados_treinamento, method="svmRadial", metric=metric, preProc=c("center", "scale"), trControl=control, fit=FALSE)
set.seed(seed)
model_svm2 <- train(Class~., data=dados_treinamento, method="svmRadialWeights", metric=metric, preProc=c("center", "scale"), trControl=control, fit=FALSE)
set.seed(seed)
model_svm3 <- train(Class~., data=dados_treinamento, method="svmPoly", metric=metric, preProc=c("center", "scale"), trControl=control, fit=FALSE)
############################## RANDOM FOREST ############################## 
set.seed(seed)
model_rf1 <- train(Class~., data=dados_treinamento, method="rf", metric=metric, trControl=control)
set.seed(seed)
model_rf2 <- train(Class~., data=dados_treinamento, method="cforest", metric=metric, trControl=control)
set.seed(seed)
model_rf3 <- train(Class~., data=dados_treinamento, method="parRF", metric=metric, trControl=control)
############################## C5.0 ############################## 
set.seed(seed)
model_c50_1 <- train(Class~., data=dados_treinamento, method="C5.0", metric=metric, trControl=control)
set.seed(seed)
model_c50_2<- train(Class~., data=dados_treinamento, method="C5.0Rules", metric=metric, trControl=control)
set.seed(seed)
model_c50_3 <- train(Class~., data=dados_treinamento, method="C5.0Tree", metric=metric, trControl=control)
############################## KNN ############################## 
set.seed(seed)
model_knn1 <- train(Class~., data=dados_treinamento, method="knn", metric=metric, preProc=c("center", "scale"), trControl=control)
set.seed(seed)
model_knn2 <- train(Class~., data=dados_treinamento, method="kknn", metric=metric, preProc=c("center", "scale"), trControl=control)
############################## RPART ############################## 
set.seed(seed)
model_cart1 <- train(Class~., data=dados_treinamento, method="rpart", metric=metric, trControl=control)
set.seed(seed)
model_cart2 <- train(Class~., data=dados_treinamento, method="rpart1SE", metric=metric, trControl=control)
set.seed(seed)
model_cart3 <- train(Class~., data=dados_treinamento, method="rpart2", metric=metric, trControl=control)
############################## RNA ##############################
set.seed(seed)
model_rna1 <- train(Class~., data=dados_treinamento, trControl=control,metric=metric, method="nnet", preProcess = c("center","scale"))
set.seed(seed)
model_rna2 <- train(Class~., data=dados_treinamento, trControl=control,metric=metric, method="mlp", preProcess = c("center","scale"))
set.seed(seed)
model_rna3 <- train(Class~., data=dados_treinamento, trControl=control,metric=metric, method="mlpML", preProcess = c("center","scale"))

####################################  K-MEANS  #################################### 

#### M??todo de Elbow ####
set.seed(seed)
x = fviz_nbclust(dados_treinamento_sem_label_scaled, kmeans, method = "wss")+
  geom_vline(xintercept = 6, linetype = 2)
x$data$clusters = 1:10
x$labels$y <- "Soma dos quadrados intra-clusters"
x$labels$x <- "Quantidade de clusters"
x$labels$title <- ""
x
### Modelo K-means ###
set.seed(seed)
model_kmeans6 <- kmeans(dados_treinamento_sem_label_scaled,6,nstart=25)
model_kmeans6
# Tabela de clusters (treinamento)
kmeans_tab=table(predictions = model_kmeans6$cluster, actual = dados_treinamento$Class)
# Plot dos clusters
fviz_cluster(model_kmeans6, geom = "point",  data = dados_treinamento_sem_label_scaled) + ggtitle("Clusters = 6")

kmeans_tab
  
#summary(dados_treinamento$Class)
####

results <- resamples(list("SVM 1"=model_svm1,"SVM 2"=model_svm2, "SVM 3"= model_svm3, "k-NN 1"=model_knn1, "k-NN 2" = model_knn2, 
                          "Naive Bayes 1"=model_nbayes1, "Naive Bayes 2" = model_nbayes2, 
                          "CART 1"=model_cart1, "CART 2"=model_cart2, "CART 3"=model_cart3, "C5.0 1"=model_c50_1,
                          "C5.0 2" = model_c50_2, "C5.0 3"= model_c50_3,"Random Forest 1"=model_rf1,"Random Forest 2"=model_rf2,"Random Forest 3"=model_rf3, 
                          "RNA 1"=model_rna1, "RNA 2" = model_rna2, "RNA 3"=model_rna3)) 


# Compara????o
summary(results)

# Boxplots dos resultados
accuracy_training_models=bwplot(results,pch='|')
plot(accuracy_training_models[1])

########################## RESULTADOS ########################## 

######################### Winner models ######################### 
######## RNA
model_rna1
model_rna1$results
predicao_rna1 <- predict(model_rna1, dados_teste_sem_label)
t<-table(predictions = predicao_rna1, actual = labels_dados_teste)
t
prop.table(table(predicao_rna1 == labels_dados_teste))
######## SVM
model_svm3
model_svm3$results
predicao_svm3 <- predict(model_svm3, dados_teste_sem_label)
t<-table(predictions = predicao_svm3, actual = labels_dados_teste)
t
prop.table(table(predicao_svm3 == labels_dados_teste))
####### C5.0
model_c50_1
model_c50_1$results
predicao_c50_1 <- predict(model_c50_1, dados_teste_sem_label)
t<-table(predictions = predicao_c50_1, actual = labels_dados_teste)
t
prop.table(table(predicao_c50_1 == labels_dados_teste))
###### RF
model_rf3
model_rf3$results
predicao_rf3 <- predict(model_rf3, dados_teste_sem_label)
t<-table(predictions = predicao_rf3, actual = labels_dados_teste)
t
prop.table(table(predicao_rf3 == labels_dados_teste))
###### K-NN
model_knn1
model_knn1$results
predicao_knn1 <- predict(model_knn1, dados_teste_sem_label)
t<-table(predictions = predicao_knn1, actual = labels_dados_teste)
t
prop.table(table(predicao_knn1 == labels_dados_teste))
###### CART
model_cart2
model_cart2$results
predicao_rpart2 <- predict(model_cart2, dados_teste_sem_label)
t<-table(predictions = predicao_rpart2, actual = labels_dados_teste)
t
prop.table(table(predicao_rpart2 == labels_dados_teste))
###### NAIVE BAYES
model_nbayes2
model_nbayes2$results
predicao_nbayes2 <- predict(model_nbayes2, dados_teste_sem_label)
t<-table(predictions = predicao_nbayes2, actual = labels_dados_teste)
t
prop.table(table(predicao_nbayes2 == labels_dados_teste))

##### K-MEANS
predicao_kmeans6 <- predict(model_kmeans6, dados_teste_sem_label_scaled)
t<-table(predictions = predicao_kmeans6, actual = labels_dados_teste)
t

classes=define_clusters(model_kmeans6)

bus_ob <- 0
opel_ob <- 0
saab_ob <- 0
van_ob <- 0

for (i in 1:length(classes)){
  if(classes[i] == '1'){
    bus_ob <- bus_ob + t[i,]
  }
  else if (classes[i] == '2'){
    opel_ob <- opel_ob + t[i,]
  }
  else if (classes[i] == '3'){
    saab_ob <- saab_ob + t[i,]  
  }
  else if (classes[i] == '4'){
    van_ob <- van_ob + t[i,]    
  }
}

# Matriz de confus??o do k-means
cfm_kmeans <- predicao_kmeans6
# Tabela/Matriz de confus??o do kmeans considerando classes 1:bus,2:opel,3:saab,4:van
new_table_kmeans <- matrix(c(bus_ob,opel_ob,saab_ob,van_ob),nrow = 4,ncol = 4,byrow=TRUE)
# Calcula sensibilidade, especificidade e eficiencia, nesta ordem, para a matriz de confus??o
return_metrics(new_table_kmeans,"bus")
return_metrics(new_table_kmeans,"opel")
return_metrics(new_table_kmeans,"saab")
return_metrics(new_table_kmeans,"van")

#################### Matriz de confus??o ####################
cfm_rna1 <- confusionMatrix(predicao_rna1, labels_dados_teste)
cfm_svm3 <- confusionMatrix(predicao_svm3, labels_dados_teste)
cfm_c50_1 <- confusionMatrix(predicao_c50_1, labels_dados_teste)
cfm_rf3 <- confusionMatrix(predicao_rf3, labels_dados_teste)
cfm_knn1 <- confusionMatrix(predicao_knn1, labels_dados_teste)
cfm_rpart2 <- confusionMatrix(predicao_rpart2, labels_dados_teste)
cfm_nbayes2 <- confusionMatrix(predicao_nbayes2, labels_dados_teste)

cfm_rna1
cfm_svm3
cfm_c50_1
cfm_rf3
cfm_knn1
cfm_rpart2
cfm_nbayes2

