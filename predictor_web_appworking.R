library(shiny)
library(shinythemes)
install.packages('rsconnect')
rsconnect::setAccountInfo(name='kaneaug',
                          token='B8DBE633A4B3BF9525998152ED8D0430',
                          secret='55Vf87jFTSgVACHyldx6ULX10gP6Qqu+EtNb7sLP')
library(rsconnect)

# Load insurance dataset
data <- read.csv("C:/Users/kanem/Documents/ML_insurance_cost_pre/data/insurance.csv")

predict_insurance_premium <- function(age, sex, bmi, children, smoker, region, 
                                      coef = c(251.40512196, 26.11715966, 330.64637157, 580.27438296, -23928.10171061, 212.22242728),
                                      intercept = 16109.88748508182) {
  
  # Convert categorical variables to binary
  if (sex == "Male") {
    sex_male <- 1
    sex_female <- 0
  } else {
    sex_male <- 0
    sex_female <- 1
  }
  
  if (smoker == "Yes") {
    smoker_yes <- 0
  } else {
    smoker_yes <- 1
  }
  

  
  # Calculate the prediction
  prediction <- eval(intercept + 
                      (coef[1] * as.numeric(age)) + 
                      (coef[2] * sex_male) + 
                      (coef[3] * as.numeric(bmi)) + 
                      (coef[4] * as.numeric(children)) + 
                      (coef[5] * smoker_yes))
                      
  return(prediction)
}




# Define UI

ui <- fluidPage(theme = shinytheme("sandstone"),
                tags$head(tags$style(HTML('
    body {
      position: relative;
    }
    body::before {
      content: "";
      background-image: url("https://pix4free.org/assets/library/2021-10-13/originals/health-insurance.jpg");
      opacity: 0.25;
      background-size: cover;
      background-position: center center;
      position: absolute;
      top: 0;
      left: 0;
      bottom: 0;
      right: 0;
      z-index: -1;
    }
  '))),
                titlePanel("Insurance Premium Predictor"),
                sidebarLayout(
                  sidebarPanel(
                    sliderInput("age", "Age", min = 18, max = 100, value = 40, step=1),
                    selectInput("sex", "Sex", choices = c("Male", "Female"), selected = "Male"),
                    sliderInput("bmi", "BMI", min = 10, max = 50, value = 30, step = 0.1),
                    selectInput("children", "Number of Children", choices = c(0, 1, 2, 3, 4, 5), selected = 1),
                    selectInput("smoker", "Smoker", choices = c("No", "Yes"), selected = "No"),
                    selectInput("region", "Region", choices = c("Northeast", "Northwest", "Southeast", "Southwest"), selected = "Northeast")
                  ),
                  mainPanel(
                    verbatimTextOutput("prediction")
                  )
                )
)


# Define server function
server <- function(input, output) {
  # Define the reactive expression
  prediction_reactive <- reactive({
    predict_insurance_premium(input$age, input$sex, input$bmi, input$children, input$smoker, input$region)
  })
  
  output$prediction <- renderText({
    paste("Predicted insurance premium: ", " $", round(prediction_reactive(), 2), sep="")
  })
}



# Run the app
shinyApp(ui, server)

