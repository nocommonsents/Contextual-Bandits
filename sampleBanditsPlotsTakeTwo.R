library(ggplot2)
library(reshape2)
all_data <- read.csv(file="banditTesting.csv")

meltdf <- melt(all_data,id="t")
ggplot(meltdf,aes(x=t,y=value,colour=variable,group=variable)) + 
    geom_line() + ggtitle('Contextual Bandit Comparison') + theme(plot.title = element_text(size=16, face="bold", vjust=2)) + 
labs(x="Trial", y=expression(paste("Cumulative Reward"))) +
theme(legend.title = element_text(size=12, face="bold"))+
  scale_color_discrete(name="Bandit Algorithm")

#labs(x="Date", y=expression(paste("Temperature ( ", degree ~ F, #" )")), title="Temperature")
#ggplot(data = all_data, aes(x=t, y=ucb_0.001)) +
#   geom_line() +
#   theme_bw()
#x <- data.frame(x = 1:100, y1 = rnorm(100), y2 = runif(100))   
#Melt into long format with sixth column as the id variable
#x.m <- melt(x, id.vars = 6)
#Plot it
#p1 <- ggplot(data = all_data, aes(x=t, y=ucb_0.001))
#p1 <- p1 + layer(geom="path") + coord_flip()
#ggplot(x.m, aes(x, value, colour = variable)) +
#  geom_line() +
#  theme_bw()

