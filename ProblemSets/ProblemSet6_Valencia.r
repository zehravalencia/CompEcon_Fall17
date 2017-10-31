
install.packages('AER',repos='http://cran.us.r-project.org')
install.packages('plm',repos='http://cran.us.r-project.org')
install.packages("haven",repos='http://cran.us.r-project.org')
install.packages("readxl",repos='http://cran.us.r-project.org')
install.packages('texreg',repos='http://cran.us.r-project.org')

library(AER)
library(plm)
library(haven)
library(readxl)
library("texreg")
library(lmtest)
library(sandwich)

## Load data and look at first 5 rows
library(foreign)
mydata <- read.dta("hospital_microold.dta") 

mydata[1:5,]
summary(mydata)

# Get a list of all column names
str(mydata)

# length of stay as dependent variable, no fixed effects 
stay1 <- lm(length_of_stay ~ lag_treatment + age +  male + hispanic + private_insurance ,data=mydata)
summary(stay1)

# length of stay as dependent variable, time fixed effects 
stay2 <- lm(length_of_stay ~ lag_treatment + age +  male + hispanic+ private_insurance + time_visit,data=mydata)
summary(stay2)

stay3 <- lm(length_of_stay ~ lag_treatment + age +  male +  hispanic + private_insurance + time_visit + faclnbr ,data=mydata)
summary(stay3)

#robust standard errors
stay1$robse <- vcovHC(stay1, type="HC1")
coeftest(stay1,stay1$robse)

stay2$robse <- vcovHC(stay2, type="HC1")
coeftest(stay2,stay2$robse)

stay3$robse <- vcovHC(stay3, type="HC1")
coeftest(stay3,stay3$robse)

print(texreg(list(stay1, stay2, stay3), dcolumn = TRUE, booktabs = TRUE,
       use.packages = FALSE, label = "tab:3", caption = "Hospital Length of Stay",
       float.pos = "hb!"))

##I had an error with memory, I could not fix it so I am printng ols results (without robust standard errors)

#install.packages('stargazer')
install.packages("stargazer", repos = "http://cran.us.r-project.org")
library(stargazer)


