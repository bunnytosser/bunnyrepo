clus<-read.csv(file='C:/Users/USUARIO/Desktop/clusterdayloss.csv') #read target excel file
library(dplyr) #Load dplyr papckage for data cleaning
dim(clus)
names(clus) #check columnnames 
clus<-clus[,8:77]


library(cluster)
library(clusterSim)


tbl_df(clus) 
clus3<-clus[,-(9:14)]
clus3<-clus3[,-c(21:26)]
clus<-clus3

t1<-lapply(clus,sd)==0
t2<-as.vector(t1)
which(t2==FALSE)
names(clus[,which(t2==T)])
clus2<-clus[,which(t2==FALSE)]
cluscentered<-scale(clus2,center=T,scale=T)

######???????????????####
library(caret)
corre<-cor(clus2)
highlycor<-findCorrelation(corre,0.6)
names(clus2[,highlycor])
[1] "SHEJIAO_3"        "SHEJIAO_PP"       "MON_FEE"          "YULE_PP"          "ZERO_TOTAL_TIMES" "ACCT_FEE"         "TOTAL_TIMES_7COE" "JF_TIMES"        
[9] "ZERO_FLUX"        "FLUX_3COE"        "XINGQU_3"         "XINGQU_P"         "TOTAL_FLUX_3AVG"  "ZHIYE_3"    
cluspear<-clus2[,-c(45,47,27,18,15,40,9)]
############
cluscentered<-scale(cluspear,center=T,scale=T)
sample1<-sample_n(as.data.frame(cluscentered),500)
sample2<-sample_n(as.data.frame(clus2),1000)
library(fpc)
library(NbClust)
library(dplyr)
library(subspace)
library(FSelector)
library(cluster)

#############forward search#########
evaluator <- function(subset) {
  k=2:10
  rst <- sapply(k, function(i){
  
    result <- kmeans(sample1, i)
    stats <- cluster.stats(dist(sample1), result$cluster)
    stats$avg.silwidth
  })
 print(names(subset))
 print(max(rst))
 return(max(rst))
}

subset <- forward.search(sample1, evaluator)
f <- as.simple.formula(subset, "Species")
print(f)


#####feature engineerng##################################


("PRODUCT_CLASS","TOTAL_FLUX_D","TOTAL_FLUX_3AVG","FLUX_3COE","ZERO_ACTIV_TIMES","ZHIYE_3","CALLED_RING","ZHIYE_PP","BIYAO_PP","XIAOFEI_PP")
c3<-c("PRODUCT_CLASS","TOTAL_FLUX_D","TOTAL_FLUX_3AVG","FLUX_3COE","ZERO_ACTIV_TIMES","ZHIYE_3",
      "CALLED_RING","ZHIYE_PP","BIYAO_PP","XIAOFEI_PP")

c2<-c(7,8,9,15,28,42,32,30,44,48)
dim(clusn1)
samplen1<-sample_n(clusn1,5000)
centeredsample<-scale(samplen1,center=T,scale=T)


#######################
samplen3<-sample_n(clusn1,8000)
centeredsample<-scale(samplen3,center=T,scale=T)
apply(centered,2,function(x) sum(is.na(x)))
c1<-c('XINGQU_P','YULE_P','SHEJIAO_PP','BIYAO_P','SHEJIAO_P','TOTAL_FLUX_3_TREND')
centered<-select(as.data.frame(centeredsample),-one_of(c1))
centered2<-select(as.data.frame(centered),-one_of(c3))
dim(centered2)


k<-3:10
rst <- sapply(k, function(i){
  result <- kmeans(centered, i)
  stats <- cluster.stats(dist(centered), result$cluster)
  stats$avg.silwidth
})
plot(k,rst,type='l',main='???????????????K?????????', ylab='????????????')
##############################


centered<-scale(clusn1,center=T,scale=T)
c1<-c('XINGQU_P','YULE_P','SHEJIAO_PP','BIYAO_P','SHEJIAO_P','TOTAL_FLUX_3_TREND')
centered<-select(as.data.frame(centered),-one_of(c1))

kmeans6<-kmeans(centered,6,nstart=10)
pk<-pamk(centered,krange=2:10,criterion='asw',usepam=FALSE,scaling=FALSE,critout=TRUE)
p2<-pam(clusn1[,1],2,diss=F,metric='euclidean',stand=T,do.swap=T)


nc <- NbClust(centered,distance='euclidean',min.nc=2,max.nc=10,method="kmeans",
              index=c('beale','silhouette','pseudot2','duda','kl','gap','db',
                      "hartigan",'ch','sdbw'))

nc <- NbClust(centered,distance='euclidean',min.nc=2,max.nc=10,method="kmeans",index=c("dindex"))

barplot(table(nc$Best.n[1,]), 
        xlab="??????", ylab="???????????????",
        main="??????7????????????????????????????????????")
#############################???????????????????????????######################
par(mfrow=c(1,1))
mydata <- scale(clusn1,center=T,scale=T)
wss <- (nrow(mydata)-1)*sum(apply(mydata,2,var))
for (i in 2:10) wss[i] <- sum(kmeans(mydata,
                                     centers=i)$withinss)
plot(1:10, wss, type="b", xlab="????????????",
     ylab="?????????????????????")

for (i in 2:10) wss[i] <- (kmeans(mydata,centers=i)$betweenss)
plot(1:10, wss, type="b", xlab="????????????",
     ylab="?????????????????????")
#####?????????####
mydatav<-sample(mydata,4000)
kv<-kmeans(mydatav,4)
mds = cmdscale(dist(mydatav,method="euclidean"),k=2)
mds = cmdscale(dist(mydatav,method="euclidean"),k=3)
plot(mds, col=kv$cluster, main='kmeans?????? k=4', pch = 19)
library(rgl)
p1<-plot3d(mds, col=kv$cluster, main='kmeans?????? k=4',size=0.8,type='s')
rgl.postscript("p1.pdf", "pdf", drawText = TRUE)

#########SOM unsupervised#######
library(kohonen)
som_grid<- somgrid(xdim=15,ydim=15,topo = "hexagonal")
plot(som_grid)
system.time(som_model2 <- som(mydata, 
                             grid=som_grid, 
                             rlen=200, 
                             alpha=c(0.05,0.01), 
                             n.hood = "circular",
                             keep.data = TRUE ))


#######PAM clustering#####3
mydata <- scale(clusn1,center=T,scale=T)
mydata1<-sample(mydata,8000)
pamk.best<-pamk(mydata1,krange=2:10,scaling=FALSE,diss=FALSE)
cat("number of clusters estimated by optimum average silhouette width:", pamk.best$nc, "\n")
mydatapam<-sample_n(clusn1,30000)
pam9<-pam(mydatapam,10,diss=FALSE,metric='euclidean',stand=TRUE)


plot(k,rst,type='l',main='???????????????K?????????', ylab='????????????')dim(clusc) # plot silhouettes
k<-kmeans(clusc,4)


k<-k2$centers
mat<-t(k)
colnames(mat)<-c('???1','???2','???3')
write.csv(mat,file='C:/Users/USUARIO/Desktop/clusteranalysisresults.csv') 
q<-ggplot(data=as.data.frame(mds))
q+geom_point(aes(x=mds[,1],y=mds[,2],color=as.factor(ksample$cluster),shape=as.factor(ksample$cluster)),
             alpha=0.4)+xlab('????????????')+ylab('????????????')+ggtitle('???????????????????????????')+theme(legend.position=c(.92,.9))