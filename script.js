// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});

// Add active class to navigation links on scroll
window.addEventListener('scroll', () => {
    let current = '';
    const sections = document.querySelectorAll('section');
    const navLinks = document.querySelectorAll('.nav-links a');

    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.clientHeight;
        if (pageYOffset >= sectionTop - 60) {
            current = section.getAttribute('id');
        }
    });

    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href').slice(1) === current) {
            link.classList.add('active');
        }
    });
});

// Project modal data updated with all projects
const projectData = {
    office1: {
        img: 'assets/project-office1.png',
        title: 'Market Mix Modeling for Optimized Budget Allocation',
        desc: 'Prepared end-to-end solutions in Azure ML Studio for market-mix modeling and dynamic pricing for price optimization.',
        tags: ['Azure ML Studio', 'Regression',"Retail-CPG", "Python", "MLflow"],
        details: `
- **Objective**:
    - Developed a comprehensive market mix model to optimize marketing budget allocation across multiple channels
    - Aimed to maximize ROI and improve marketing efficiency by identifying the most effective channels

- **Key Technologies & Tools Used**:
    - Azure ML Studio for model development and deployment
    - Python for data preprocessing and analysis
    - Regression analysis techniques
    - Statistical modeling tools

- **Approach/Methodology**:
    1. Data Collection: Gathered historical marketing spend and performance data across channels
    2. Feature Engineering: Created relevant variables to capture marketing impact
    3. Model Development: Built regression models to quantify channel effectiveness
    4. Validation: Tested model accuracy and robustness
    5. Implementation: Deployed solution in Azure ML Studio

- **Outcome/Results**:
    - 25% improvement in marketing ROI
    - 30% reduction in inefficient spend
    - Better understanding of channel effectiveness
    - Data-driven budget allocation decisions

- **Challenges & Solutions**:
    - Data Quality: Implemented robust data cleaning and validation
    - Model Complexity: Used stepwise regression for feature selection
    - Integration: Created seamless Azure ML Studio workflow

- **Impact**:
    The market mix model transformed how marketing budgets were allocated, leading to significant cost savings and improved campaign performance. The solution provided actionable insights for marketing strategy optimization.`
    },
    office2: {
        img: 'assets/project-office2.png',
        title: 'Dynamic Pricing for Optimizing Pricing Strategy',
        desc: 'Implemented dynamic pricing models to optimize product pricing strategies, increasing revenue and market competitiveness.',
        tags: ['Azure ML Studio', 'Regression',"Retail-CPG",'MLflow', "Python"],
        details: `
####  **Objective**:
    
This project was designed as a hands-on learning experience by the Enqurious team, focusing on developing and deploying dynamic pricing strategies. The goal was to optimize pricing decisions for GlobalMart's detergent brand, Tide, to increase revenue and maintain competitive positioning amid market pressures. By analyzing sales, competitor, customer behavior, and inventory data, the project aimed to maximize revenue while balancing demand and inventory efficiency.
    
#### **Key Technologies & Tools Used**:
    
- **Programming Languages & Libraries:**  
        Python (pandas for data manipulation, statsmodels for regression modeling, scipy for optimization, matplotlib for visualization).
        
 - **Cloud Platform:**  
        Azure ML Studio was leveraged for end-to-end model building, tracking, versioning via MLflow, and deployment including inference endpoints.
        
- **Techniques:**  
        Marketing Mix Modeling, log-log and polynomial regression models to capture elasticity and nonlinear effects, L1 (Lasso) and L2 (Ridge) regularization to prevent overfitting, feature engineering to create meaningful predictors from sales and pricing data, and optimization techniques to simulate pricing scenarios.
        
####  **Approach/Methodology**:
    
- Integrated and preprocessed multiple datasets encompassing sales, pricing, competitor prices, customer behavior, and inventory data.
        
- Performed Exploratory Data Analysis (EDA) to identify patterns, seasonality, and key factors affecting demand for Tide detergent.
        
- Engineered features such as demand elasticity, competitor pricing influence, temporal variables, and inventory levels.
        
- Developed regression models (log-log and polynomial) with regularization (L1 and L2) to predict demand and optimize pricing strategy.
        
- Validated models using cross-validation and performance metrics (R-squared, RMSE).
        
- Implemented model lifecycle management with MLflow for reproducibility and version control.
        
- Automated deployment and monitoring through Azure ML Studio's CI/CD pipelines and inference endpoints.
        
#### **Outcome/Results**:
    
- Delivered a robust dynamic pricing model that successfully predicted demand sensitivity to price changes.
        
- Optimized Tide's pricing strategy to increase revenue while minimizing inventory inefficiencies and maintaining customer demand.
        
- Enabled GlobalMart to respond proactively to competitor pricing actions and market shifts.
        
- Facilitated data-driven pricing adjustments improving both profitability and customer satisfaction.
        
#### **Challenges & Solutions**:
    
- **Challenge:** Complex integration of diverse datasets with missing and noisy data.  
        **Solution:** Applied thorough data cleaning, normalization, and feature engineering to create reliable predictors.
        
- **Challenge:** Capturing nonlinear and delayed effects of price changes on demand.  
        **Solution:** Employed polynomial regression and lagged features, alongside L1/L2 regularization to improve model generalization.
        
- **Challenge:** Ensuring model reproducibility and smooth deployment in a production environment.  
        **Solution:** Leveraged MLflow for model tracking/versioning and Azure ML Studio's CI/CD pipelines for automated deployment.
        
#### **Impact**:
    
- Provided GlobalMart with an adaptive pricing tool that increased profit margins and competitive edge in the detergent market.
        
- Reduced revenue leakage due to static pricing by enabling dynamic, data-driven price adjustments.
        
- Supported continuous improvement with a scalable model deployment and monitoring framework.
        
- Empowered the business to make strategic decisions informed by robust machine learning insights.`
    },
    office3: {
        img: 'assets/project-office3.png',
        title: 'Predicting Optimal Channel for New Product Launch',
        desc: 'Built predictive models to identify the most effective marketing channels for launching new products, improving launch success rates.',
        tags: ['Python', 'Classification', 'Azure ML Studio', 'MLflow', 'Retail-CPG'],
        details: `
#### **Objective**:
    
This project was prepared as a hands-on data science project designed for learning and skill development, serving the educational goals of Enqurious. The primary goal is to predict the optimal marketing and distribution channel for launching GlobalBev's innovative new beverage line. By leveraging historical sales, customer preferences, and competitive data, the project aims to identify the channel(s) that maximize reach and profitability, avoiding a broad but inefficient launch.

#### **Key Technologies & Tools Used**:

- Predictive analytics and machine learning for modeling channel performance.
    
- Data cleaning and aggregation techniques.
    
- Exploratory Data Analysis (EDA) tools for insights extraction.
    
- Azure ML Studio for building, training, and deploying machine learning models.
    
- MLflow for model tracking, versioning, and reproducibility.
    
- CI/CD pipelines to automate model deployment.
    
- Production deployment tools for inference endpoints.
    
- Marketing and distribution channel data sources (e-commerce, Q-commerce, modern trade, general trade, HoReCa).

#### **Approach/Methodology**:

- Clean, aggregate, and normalize historical sales, customer preferences, and competitive data across channels.
    
- Perform EDA to identify key performance indicators and regional customer preferences.
    
- Derive composite performance scores for each channel based on raw metrics like units sold, customer preference, and competitive pressure.
    
- Develop and evaluate machine learning models to predict the optimal channel for product launch.
    
- Integrate MLflow for lifecycle management and implement CI/CD pipelines for seamless deployment.

#### **Outcome/Results**:
    
- Delivered a predictive model that accurately identified the best channel(s) for product launch with high confidence.
        
- Enabled GlobalBev to strategically focus marketing and distribution efforts, reducing wasted resources.
        
- Provided actionable insights into regional customer preferences and competitive dynamics.
        
#### **Challenges & Solutions**:
    
- Challenge: Gathering and normalizing diverse channel data with varying formats and granularity.  
        Solution: Developed robust data cleaning and aggregation pipelines to create a unified dataset.
        
- Challenge: Identifying which features would best train the model and preparing the data accordingly required significant learning and effort.  
        Solution: Conducted thorough feature engineering and exploratory data analysis to select impactful variables, improving model accuracy.
        
- Challenge: Ensuring reproducibility and smooth deployment.  
        Solution: Used MLflow and CI/CD pipelines to automate model tracking, versioning, and deployment.
        
#### **Impact**:
    
- Helped GlobalBev launch its new beverage with a targeted channel strategy, leading to optimized resource allocation.
        
- Increased confidence in channel selection, resulting in higher market penetration and profitability.
        
- Set a foundation for continuous improvement through productionized machine learning models and monitoring.`
    },
    office4: {
        img: 'assets/project-office4.png',
        title: 'Proof Of Concept : AI Agent for Content Inventory Exploration',
        desc: 'Developed a proof-of-concept AI agent to help the Content Development Team explore and analyze content inventory efficiently.',
        tags: ['AI Agent', 'Content Team', "Python", "Langchain", "Streamlit", "Azure SQl"],
        details: `
#### **Objective**:

The primary goal of creating this AI agent was to automate and simplify the exploration and management of the extensive content inventory maintained by the Enqurious content team. With hundreds of content pieces produced regularly, it became difficult to keep track of published items, especially those needing fixes or ready to be showcased to clients. The agent was designed to reduce manual effort, speed up content discovery, and support creating a clean catalog of deliverable content.

#### **Key Technologies & Tools Used**:

- **AI and NLP**: Utilized a large language model (Claude 3.7 Sonnet) for natural language understanding and response generation.
    
- **Langchain framework**: Enabled building AI applications and agents with ease.
    
- **Streamlit**: Provided a user-friendly chat-like interface for interaction with the agent.
    
- **Azure SQL Database**: Source of the content data.
    
- **Google Sheets & CSV files**: Used for data visibility and to simplify the proof-of-concept setup.
    
- **Python libraries**: .env for secure key management, Langchain-anthropic for AI integration.

#### **Approach/Methodology**:

- **Understanding the Problem**: Identified the pain points of manual content inventory checks and inconsistent tracking.
    
- **Data Preparation**: Data was extracted from Azure SQL to Google Sheets and downloaded as CSV files. Data cleaning involved fixing inconsistent column names, formatting, and removing errors.
    
- **Organizing Data**: Content was partitioned into logical folders by creator and content status (draft, published, archived) to simplify data access.
    
- **Building the Agent**: Developed a CSV agent using Langchain connected to the cleaned CSV files. The agent could perform SQL-like queries based on user text prompts.
    
- **Creating a UI**: Built a Streamlit app to simulate a chat interface for natural user interaction.
    
- **Prompt Engineering**: Designed prompts to specify content creator, status, and query type, ensuring relevant and formatted output.
    
- **Testing and Iteration**: Tested with real user queries, identified inconsistencies and performance issues, and outlined future improvements.

#### **Outcome/Results**:

- Significant reduction in manual effort needed to find, filter, and verify content status.
    
- Faster content discovery and generation of reports in a consistent format.
    
- Improved accuracy in identifying content ready for client delivery.
    
- User-friendly interface enabled easy adoption by content developers without technical expertise.
    
- Modular design prepared the system for scaling and integration with cloud infrastructure.

#### **Challenges & Solutions**:

- **Challenge**: Data Quality Issues - Messy data with inconsistent formats and missing entries initially hindered performance.
    
    - **Solution**: Addressed by thorough data cleaning and standardized formatting.
    
- **Challenge**: Response Inconsistency - AI agent sometimes returned answers in varying formats (tables, bullets, paragraphs), which complicated report generation.
    
    - **Solution**: Future work aims to enforce unified output templates.
    
- **Challenge**: Latency - Using a free AI model led to slow response times (8–10+ seconds).
    
    - **Solution**: Optimizing queries and considering paid, faster APIs.
    
- **Challenge**: Lack of Performance Metrics - No automated way to measure accuracy or user satisfaction.
    
    - **Solution**: Planned integration of feedback loops and monitoring metrics.

#### **Impact**:

This AI agent fundamentally transformed how the content development team interacted with their content inventory. It provided an intelligent, automated assistant capable of swiftly answering questions and generating reports, thus saving time and reducing human error. By automating content exploration, the agent improved the quality and timeliness of client-ready content catalogs and empowered the team to focus more on content creation rather than inventory management. The approach laid a solid foundation for scaling content operations and integrating advanced AI-driven workflows in the future.`
    },
    office5: {
        img: 'assets/project-office5.png',
        title: 'POC: AI Agent for Reporting to Client',
        desc: 'Created a proof-of-concept AI agent for the Ops Team to automate and streamline client reporting processes.',
        tags: ['AI Agent', 'Ops Team', "Python", "Langchain", "Streamlit", "Azure SQl"],
        details: `
#### **Objective**:
    
The AI Agent is designed to streamline and automate responses to operational and tactical queries from both internal and external stakeholders, significantly reducing manual effort in tracking and reporting. Its primary goal is to enable real-time, accurate, and role-based access to critical insights from database systems, facilitating faster decision-making and enhancing communication with clients and internal teams.
    
#### **Key Technologies & Tools Used**:
    
- **AI Models & Multi-Agent Architecture:** A hierarchical multi-agent system comprising an Orchestrator Agent (for query intent detection and routing), Engagement Agent (tracking learner engagement and progress), and Performance Agent (monitoring assessment lifecycle and certification status).
        
- **Data Processing & Storage:** Utilizes Azure Data Lake Storage (ADLS) with a structured pipeline involving Bronze, Silver, and Gold layers for refined data preparation.
        
- **Batch Inference & Liquid Clustering:** Performs batch inference twice daily on partitioned data clusters (based on client ID, program ID, and table type) to ensure scalable and efficient processing.
        
- **Reporting Automation:** Generates customizable CSV/PDF reports on-demand based on user queries.
        
- **Security & Access Control:** Implements role-based access control (RBAC) to safeguard sensitive data and ensure users receive data appropriate to their access levels.
        
#### **Approach/Methodology**:
    
- Data flows from ADLS through Bronze, Silver, and Gold layers, where Liquid Clustering partitions data to improve query efficiency.
        
- The Orchestrator Agent interprets the user query's intent and routes it to appropriate sub-agents—Engagement or Performance—running in parallel to process relevant data.
        
- Users (internal ops teams like Prateek, Abhilipsa, Chetan; monitoring team Amit, Soma; mentors; and external clients/managers) submit text-based queries, often with templates or FAQs, allowing non-technical users to interact via natural language.
        
- The system filters queries based on security level and user roles before executing batch inference to provide structured or textual responses in real time.
        
- Reporting tools convert query results into preferred formats such as CSV or PDF, delivered instantly to users.
        
#### **Outcome/Results**:
    
- The AI Agent reduced report generation time from hours of manual effort to seconds, significantly improving operational efficiency.
        
- Enabled non-technical users to obtain accurate, tailored insights without needing SQL knowledge, enhancing accessibility and user experience.
        
- Improved decision-making speed by providing timely, actionable information on learner engagement, assessment progress, and certification status.
        
- Minimized repetitive manual queries and reporting overhead for the operations team.
        
#### **Challenges & Solutions**:
    
- **Challenge:** Handling complex, diverse queries from users with varying technical skills.  
        **Solution:** Designed a multi-agent system with intent classification and query routing to specialized agents for accurate, context-aware responses. Provided predefined query templates for common requests to assist non-technical users.
        
- **Challenge:** Ensuring data security and role-based access across internal and external stakeholders.  
        **Solution:** Implemented strict RBAC and query-level security filters to prevent unauthorized data exposure.
        
- **Challenge:** Processing large volumes of data efficiently while maintaining real-time responsiveness.  
        **Solution:** Adopted batch inference with liquid clustering to partition data and optimize compute resources.
        
- **Challenge:** Preventing hallucinated or incorrect AI responses that could mislead users.  
        **Solution:** Established response validation rules and fallback mechanisms to ensure accuracy and reliability.
        
#### **Impact**:
    
- Transformed the reporting workflow by automating complex tracking and reporting tasks, freeing the operations team to focus on higher-value activities.
        
- Enhanced transparency and communication with clients and internal stakeholders through instant, precise reporting.
        
- Established a scalable and secure AI-powered reporting framework adaptable to future expansions and additional data sources.`
    },
    self1: {
        img: 'assets/project-self1.png',
        title: 'Market Mix Modeling',
        desc: 'Built a market mix model to analyze and optimize marketing spend across multiple channels for improved ROI.',
        tags: ['Python', 'Retail-CPG', "MLflow", "Azure ML Studio", "Regression"],
        details: `
#### **Objective**:

To improve marketing ROI for a global hair care brand by optimizing the allocation of marketing budget across multiple paid and organic channels. The primary goal was to identify which media investments (like Paid Social, Paid Search, Email, and Modular Video) were truly contributing to sales, and reallocate budgets accordingly using statistical and machine learning modeling.

#### **Key Technologies & Tools Used**:

- **Tools**: Python (pandas, statsmodels, scipy, matplotlib), Excel
    
- **Techniques**: Marketing Mix Modeling (MMM), Linear and Log-Log Regression, Hill Function for saturation modeling, Adstock transformation, Multicollinearity diagnosis (VIF), Optimization via simulation

#### **Approach/Methodology**:

- **Data Preparation**: Cleaned and standardized 122 weeks of historical marketing and sales data. This included handling Indian-number formats, missing values, and encoding of categorical features like holidays.
    
- **Feature Engineering**: Created adstocked and saturation-adjusted versions of media spend, extracted temporal features (month, week, year), and lagged variables (e.g., gasoline price) to reflect behavioral delays.
    
- **Modeling**:
    
    - Built both additive and log-log OLS regression models to estimate the impact (elasticities) of each marketing channel.
        
    - Included control variables (e.g., price, SKUs, gasoline prices, holidays) to isolate true media effects.
        
    - Performed multicollinearity checks and reduced model complexity by dropping redundant variables and interaction terms.
        
- **Optimization**: Simulated budget reallocation scenarios using the \`scipy.optimize.minimize\` function, constrained by historical ranges and statistical significance.

#### **Outcome/Results**:

- The **log-log model explained ~84% of the variance in sales**, with **organic search emerging as the most effective sales driver**.
    
- **Paid Search** showed strong ROI despite low absolute spend and was recommended for scale-up.
    
- **Email and Paid Social** demonstrated statistically negative or insignificant effects, suggesting overinvestment or campaign fatigue.
    
- The final optimization suggested:
    
    - **Eliminating spend** on Email, Paid Social, and Modular Video due to poor or negative ROI.
        
    - **Focusing on Paid Search** for immediate ROI and **modular video** for long-term brand building (with caution).
        
- Provided clear budget reallocation recommendations grounded in model elasticities and saturation thresholds.

#### **Challenges & Solutions**:

- **Challenge**: Severe multicollinearity across media variables due to overlapping representations (adstock, saturation, interactions).
    
    - **Solution**: Dropped log-adstock variables and retained only one representation per media channel (preferably saturation-adjusted).
        
- **Challenge**: Some channels (e.g., Paid Social, Email) showed theoretically high ROI but weren't statistically significant.
    
    - **Solution**: Applied coefficient significance weighting and constrained optimization to avoid over-relying on noisy variables.
        
- **Challenge**: Capturing delayed or lingering effects of media spend.
    
    - **Solution**: Used adstock transformation with appropriate decay rates (e.g., 0.6 for Email, 0.9 for Modular Video) to model carryover impact.

#### **Impact**:

- Demonstrated how MMM can lead to **data-driven marketing strategy**, with clear ROI accountability.
    
- Enabled **reduction of ineffective spend** and highlighted opportunities for **scalable growth in efficient channels**.
    
- The approach is applicable for real-world marketing teams to simulate "what-if" scenarios and optimize media allocation under budget constraints.
    
- Though part of an assignment, this project represents industry-grade marketing analytics and strategic budget planning.`
    },
    self2: {
        img: 'assets/project-self2.png',
        title: 'Water Potability Prediction',
        desc: 'Developed a scoring system to profile water quality using data analysis and machine learning techniques.',
        tags: ['Python', "Regression", "EDA", "Feature Engineering"],
        details: `
#### **Objective**:

To develop a regression model to predict the **potability** of water based on its chemical characteristics such as pH, hardness, solids, and contaminants like chloramines and trihalomethanes. The aim was to build a predictive system that could guide water quality assessment using easily measurable features.

#### **Key Technologies & Tools Used**:

- **Tools**: Python (pandas, scikit-learn, XGBoost, matplotlib, seaborn, numpy)
    
- **Techniques**:
    
    - Data cleaning and preprocessing
        
    - Feature engineering
        
    - Model selection and evaluation
        
    - Regression modeling (Linear Regression, Random Forest, Gradient Boosting)
        
    - Custom metric optimization (based on MAE)
        


#### **Approach/Methodology**:

- **Problem Framing**: Defined the problem as a regression task due to the continuous nature of the potability target variable.
    
- **Data Cleaning**:
    
    - Addressed missing values
        
    - Removed text artifacts and HTML tags from numerical columns
        
    - Converted object-type numeric fields into clean float formats
        
- **Feature Engineering**:
    
    - Created additional features such as deviation from optimal pH
        
    - Examined chemical ratios and possible interaction terms
        
- **Modeling**:
    
    - Started with Linear Regression as a baseline
        
    - Advanced to ensemble models like Random Forest and Gradient Boosting
        
    - Evaluated performance using MAE and a custom score: \`Score = max(0, 100 * (1 - MAE))\`
        
- **Model Tuning**: Optimized hyperparameters using cross-validation techniques.
    


#### **Outcome/Results**:

- **Best performing models** were ensemble-based regressors (Random Forest and Gradient Boosting).
    
- The model achieved **high predictive accuracy** under the custom evaluation metric, with significantly reduced Mean Absolute Error compared to the baseline.
    
- **Key contributing features** included pH levels, trihalomethane concentrations, and organic carbon content.
    
- Insights derived could help inform water treatment interventions based on chemical makeup.
    


#### **Challenges & Solutions**:

- **Challenge**: Many columns were improperly typed (as strings) with non-numeric values.
    
    - **Solution**: Created robust data-cleaning functions to convert these into usable numerical formats.
        
- **Challenge**: Handling missing data in critical features.
    
    - **Solution**: Used statistical imputation strategies (mean/median), guided by domain relevance.
        
- **Challenge**: Identifying which model best fit the complex relationships in the data.
    
    - **Solution**: Applied multiple regression algorithms and compared their MAE and final scores to select the optimal model.
        


#### **Impact**:

- Demonstrated how regression modeling can be used to support **environmental safety decisions**.
    
- This pipeline could be adapted to real-time water monitoring systems or **deployed in IoT sensors** for smart water quality management.
    
- The project shows practical **ML application in public health**, especially where expert chemical analysis might be limited or delayed.`
    },
    self3: {
        img: 'assets/project-self3.png',
        title: 'Book Recommendation (RAG Chain)',
        desc: 'Implemented a book recommendation system using Retrieval-Augmented Generation (RAG) and NLP techniques.',
        tags: ['Python', 'RAG', 'NLP', "Streamlit", "Pinecone", "Langchain", "Hugging Face", "Ollama"],
        details: `
#### **Objective**:

To develop an AI-powered chatbot that provides personalized book recommendations using **natural language processing (NLP)** and **retrieval-augmented generation (RAG)**. The goal was to build a conversational assistant that understands user queries, retrieves relevant book metadata using vector similarity, and generates human-like recommendations.



#### **Key Technologies & Tools Used**:

- **Tools**: Python, Streamlit, Hugging Face Transformers, Pinecone, LangChain, Ollama (Mistral)
    
- **Libraries**: Pandas, NumPy, NLTK, SpaCy, SentenceTransformers
    
- **Techniques**: Text preprocessing, embeddings, query classification, vector similarity search, conversational AI with RAG
    



#### **Approach/Methodology**:

- **Phase 1 – Data Collection & Cleaning**:
    
    - Used a Kaggle dataset of books with metadata (title, author, genres, description, ratings).
        
    - Cleaned HTML tags, stopwords, malformed URLs, and standardized formats (ratings, dates).
        
- **Phase 2 – Data Enhancement**:
    
    - Created a unified semantic field by combining relevant metadata for embedding.
        
    - Filtered out short/incomplete descriptions and normalized genre categories.
        
- **Phase 3 – Embedding Generation & Vector Search**:
    
    - Used the \`all-MiniLM-L6-v2\` transformer to generate 512-dimensional embeddings.
        
    - Stored embeddings in **Pinecone** to perform fast similarity-based retrieval.
        
- **Phase 4 – RAG Pipeline & Chat Interface**:
    
    - Transformed user queries into embeddings, performed vector search, and passed results to **Ollama** (Mistral) for conversational responses.
        
    - Built a chatbot interface using **Streamlit**, supporting both simple and conversational modes.
        



#### **Outcome/Results**:

- Two chatbot versions were developed:
    
    1. **Simple Bot**: Fast, direct book recommendations without conversation history.
        
    2. **Conversational Bot**: Dynamic interactions with query classification, chat memory, and structured prompts.
        
- Delivered highly relevant book suggestions based on semantic understanding of user queries.
    
- Supported multiple query types (recommendations, genre discussion, summaries, etc.).
    
- Demonstrated scalable architecture using vector search and AI-driven generation.
    



#### **Challenges & Solutions**:

- **Challenge**: Raw book metadata was noisy and inconsistent.
    
    - **Solution**: Implemented robust data cleaning and transformation pipelines (Bronze → Silver → Gold layers).
        
- **Challenge**: Capturing semantic similarity across diverse book descriptions.
    
    - **Solution**: Used domain-appropriate embedding models (MiniLM) and refined text fields for accuracy.
        
- **Challenge**: Handling different user intents in conversation.
    
    - **Solution**: Integrated query classification to dynamically route intent types and improve interaction quality.
        



#### **Impact**:

- Created a versatile recommendation engine for:
    
    - **Readers** seeking personalized book suggestions.
        
    - **Librarians and bookstores** wanting AI-powered search tools.
        
    - **Developers and researchers** exploring RAG and semantic search applications.
        
- The chatbot architecture is modular, extensible, and deployable in both research and commercial environments.
    
- This project demonstrates practical NLP and RAG integration in building intelligent assistants that **understand, retrieve, and converse**.`
    },
    self4: {
        img: 'assets/project-self4.png',
        title: 'Customer Churn Prediction',
        desc: 'Created predictive models to identify customers at risk of churning, enabling targeted retention strategies.',
        tags: ['Python', 'Classification', "MLflow", "EDA", "Feature Engineering", "Streamlit", "Model Deployment", "Render"],
        details: `
#### **Objective**:

To predict customer churn in a subscription-based telecommunications service using machine learning techniques. The goal is to identify customers likely to leave the service so that targeted retention strategies can be implemented, improving customer lifetime value and reducing revenue loss.


#### **Key Technologies & Tools Used**:

- **Programming Languages**: Python
    
- **Tools**:
    
    - **Pandas, NumPy** (Data handling)
        
    - **Matplotlib, Seaborn** (Data visualization)
        
    - **Scikit-learn** (Modeling, preprocessing, evaluation)
        
    - **XGBoost** (Advanced modeling)
        
    - **MLflow** (Experiment tracking and model lifecycle management)
        
    - **SMOTE** (Class imbalance correction)
        
    - **Flask & Docker** (For deployment, mentioned in project pipeline)
        
- **Techniques**:
    
    - Data preprocessing (handling nulls, encoding, scaling)
        
    - Feature selection (correlation, RFE)
        
    - Model evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)
        
    - Hyperparameter tuning (GridSearchCV)
        
    - Model deployment (via MLflow and serialized \`.pkl\`)
        



#### **Approach/Methodology**:

 - **Data Understanding**:
    
    - Used the Telco Customer Churn dataset from Kaggle.
        
    - Target variable: \`Churn Label\` (binary classification).
        
 -  **Data Preprocessing**:
    
    - Cleaned missing and duplicate values.
        
    - Dropped irrelevant columns (e.g., customerID).
        
    - Converted data types (e.g., \`TotalCharges\`).
        
    - Encoded categorical variables using \`LabelEncoder\`.
        
    - Scaled numeric features using \`StandardScaler\`.
        
 -  **Feature Engineering & Selection**:
    
    - Applied correlation analysis to drop highly correlated variables.
        
    - Used **Recursive Feature Elimination (RFE)** with Random Forest to retain top features.
        
    - Handled class imbalance using **SMOTE** to synthetically balance classes.
        
 -  **Modeling**:
    
    - Built and evaluated multiple models:
        
        - Logistic Regression
            
        - Logistic Regression with L2 regularization
            
        - Random Forest (baseline and GridSearch-tuned)
            
        - XGBoost (best performer)
            
    - Tracked all experiments and metrics using **MLflow**.
        
 -  **Evaluation**:
    
    - Metrics used: Accuracy, Precision, Recall, F1 Score, ROC-AUC.
        
    - Cross-validated all models using \`cross_val_score\`.
        
 -  **Deployment**:
    
    - Serialized the best model (\`XGBoost\`) using \`pickle\`.
        
    - MLflow tracked experiment metadata and performance logs.
        
    - Mentioned future deployment using Flask and Docker.
        



#### **Outcome/Results**:

- **XGBoost** achieved the best performance with:
    
    - Accuracy: **92.15%**
        
    - AUC-ROC: **0.84**
        
- Cross-validation showed **XGBoost** had the highest average accuracy and acceptable variance.
    
- Tracked all experiments, parameters, and metrics using **MLflow**, enabling reproducibility and model comparison.
    
- Final model saved and ready for deployment (\`xgb.pkl\`).
    



#### **Challenges & Solutions**:

- **Imbalanced Classes**:
    
    - Solved using **SMOTE** to oversample the minority class.
        
- **Multicollinearity**:
    
    - Addressed by removing highly correlated features via correlation matrix.
        
- **Model Selection & Tuning**:
    
    - Used **GridSearchCV** with Random Forest to improve results.
        
    - Compared models using consistent metrics in an aggregated results dataframe.
        
- **Experiment Tracking**:
    
    - Used **MLflow** to manage and visualize experiment runs effectively.
        



#### **Impact**:

- **Business Value**: Enables proactive churn mitigation by identifying high-risk customers early.
    
- **Model Explainability**: Logistic regression results offer interpretability for stakeholders, while XGBoost ensures high accuracy.
    
- **Scalability**: Modular pipeline ready for deployment and future automation.
    
- **Reusability**: All preprocessing, modeling, and tracking components are reusable and production-ready.
    
- **Trackability**: MLflow logging ensures reproducibility and performance tracking across model versions.`
    },
    self5: {
        img: 'assets/project-self5.png',
        title: 'Customer Purchase Behavior',
        desc: 'Analyzed and modeled customer purchase patterns to uncover insights and drive business growth.',
        tags: ['Python', 'EDA', "RFM", "CLV", "Data Visualization"],
        details: `
#### **Objective**:

To identify high-value customer segments using **RFM (Recency, Frequency, Monetary)** analysis and compute **Customer Lifetime Value (CLV)**. The aim is to inform targeted marketing, optimize customer retention efforts, and enhance overall revenue by focusing on the most profitable customer groups.



#### **Key Technologies & Tools Used**:

- **Tools**: Python, Jupyter Notebook, Excel
    
- **Libraries**: pandas, numpy, datetime, matplotlib, seaborn
    
- **Techniques**:
    
    - RFM Analysis (for behavioral segmentation)
        
    - CLV Calculation (to estimate long-term customer value)
        
    - Data cleaning and feature engineering
        
    - Customer segmentation and interpretation
        



#### **Approach/Methodology**:

 - **Data Loading and Cleaning**:
    
    - Imported raw transaction data.
        
    - Removed nulls, duplicates, and invalid transactions (e.g., refunds or negative quantities).
        
    - Ensured correct date formats for accurate Recency calculation.
        
 - **RFM Analysis**:
    
    - **Recency**: Days since the customer's last purchase.
        
    - **Frequency**: Total number of purchases.
        
    - **Monetary**: Total amount spent by each customer.
        
    - Assigned scores (1–5) for each RFM metric.
        
    - Created composite RFM scores for customer segmentation.
        
 - **CLV Calculation**:
    
    - Calculated **Average Purchase Value** and **Purchase Frequency** for each customer.
        
    - Used a fixed **Customer Lifespan** assumption (e.g., 12 months) to estimate CLV:
        
        CLV=APV×PF×Lifespan\\text{CLV} = \\text{APV} \\times \\text{PF} \\times \\text{Lifespan}CLV=APV×PF×Lifespan
    - Merged CLV results with RFM segments for combined insights.
        
 - **Customer Segmentation**:
    
    - Classified customers into segments such as "Champions," "Loyal," "At Risk," and "Lost."
        
    - Visualized segments using bar plots and box plots for strategic understanding.
        



#### **Outcome/Results**:

- Identified **top 10% customers** who contributed the most to revenue.
    
- CLV distribution revealed a **small group of high-value customers** that could be prioritized for retention campaigns.
    
- RFM analysis segmented customers into actionable categories, enabling **precision targeting**.
    
- Provided **data-backed recommendations** to guide marketing campaigns and customer service strategies.
    



#### **Challenges & Solutions**:

- **Data Quality Issues**:
    
    - Resolved by filtering out refund transactions and cleaning duplicate entries.
        
- **Uneven Purchase Dates**:
    
    - Standardized the **snapshot date** for recency calculations to maintain consistency.
        
- **Lack of Customer Lifecycle Info**:
    
    - Used a reasonable **lifespan assumption** (e.g., 12 months) based on industry norms.
        



#### **Impact**:

- **Strategic Marketing**: Enabled prioritization of high-CLV and loyal segments.
    
- **Revenue Optimization**: Focused attention on customers contributing the most value.
    
- **Customer Retention**: Highlighted at-risk segments for win-back campaigns.
    
- **Scalable Framework**: Built a reusable RFM and CLV pipeline for ongoing segmentation and analysis.`
    }
};

// Project modal functionality
const modal = document.getElementById('project-modal');
const modalImg = document.querySelector('.modal-img');
const modalTitle = document.querySelector('.modal-title');
const modalClose = document.querySelector('.modal-close');
const modalDetails = document.querySelector('.modal-details');

// Configure marked options
marked.setOptions({
    breaks: true, // Enable line breaks
    gfm: true,    // Enable GitHub Flavored Markdown
    headerIds: false // Disable header IDs for better styling control
});

// Open modal on card click
document.querySelectorAll('.project-card').forEach(card => {
    card.addEventListener('click', () => {
        const key = card.getAttribute('data-project');
        const data = projectData[key];
        
        if (!data) {
            console.error('Project data not found for:', key);
            return;
        }

        // Update modal content
        modalImg.src = data.img;
        modalImg.alt = data.title;
        modalTitle.textContent = data.title;
        
        // Display detailed description if available
        if (data.details) {
            modalDetails.innerHTML = marked.parse(data.details);
            modalDetails.style.display = 'block';
        } else {
            modalDetails.style.display = 'none';
        }
        
        // Show modal
        modal.style.display = 'flex';
        modal.classList.add('active');
    });
});

// Close modal on close button or overlay click
modalClose.addEventListener('click', () => {
    modal.classList.remove('active');
    setTimeout(() => {
        modal.style.display = 'none';
    }, 300);
});

modal.addEventListener('click', e => {
    if (e.target === modal) {
        modal.classList.remove('active');
        setTimeout(() => {
            modal.style.display = 'none';
        }, 300);
    }
});

const navToggle = document.querySelector('.nav-toggle');
const navLinks = document.querySelector('.nav-links');
const navLinksItems = document.querySelectorAll('.nav-links a');

navToggle.addEventListener('click', () => {
    navLinks.classList.toggle('open');
  });

navLinksItems.forEach(link => {
    link.addEventListener('click', () => {
      navLinks.classList.remove('open');
    });
  });

// Update project tags dynamically
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.project-card').forEach(card => {
        const key = card.getAttribute('data-project');
        const data = projectData[key];
        
        if (data && data.tags) {
            const tagsContainer = card.querySelector('.project-tags');
            if (tagsContainer) {
                tagsContainer.innerHTML = data.tags.map(tag => `<span>${tag}</span>`).join('');
            }
        }
    });
});
