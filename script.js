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
        tags: ['Azure ML Studio', 'Regression'],
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
        tags: ['Azure ML Studio', 'Regression'],
        details: `
- **Objective**:
    - Developed a dynamic pricing system to optimize product pricing strategies
    - Aimed to maximize revenue while maintaining market competitiveness

- **Key Technologies & Tools Used**:
    - Azure ML Studio for model deployment
    - Python for data analysis and modeling
    - Regression analysis
    - Market research tools

- **Approach/Methodology**:
    1. Market Analysis: Studied competitor pricing and market trends
    2. Data Collection: Gathered historical sales and pricing data
    3. Model Development: Created pricing optimization algorithms
    4. Testing: Validated models with historical data
    5. Implementation: Deployed in Azure ML Studio

- **Outcome/Results**:
    - 15% increase in revenue
    - 20% improvement in market competitiveness
    - More responsive pricing strategy
    - Better understanding of price elasticity

- **Challenges & Solutions**:
    - Market Volatility: Implemented adaptive algorithms
    - Data Integration: Created automated data pipelines
    - Model Accuracy: Used ensemble methods for better predictions

- **Impact**:
    The dynamic pricing system significantly improved revenue and market position, while providing valuable insights into pricing strategies and customer behavior.`
    },
    office3: {
        img: 'assets/project-office3.png',
        title: 'Predicting Optimal Channel for New Product Launch',
        desc: 'Built predictive models to identify the most effective marketing channels for launching new products, improving launch success rates.',
        tags: ['Python', 'Classification', 'Databricks'],
        details: `
- **Objective**:
    - Developed a predictive model to identify optimal marketing channels for new product launches
    - Aimed to improve launch success rates and marketing efficiency

- **Key Technologies & Tools Used**:
    - Python for model development
    - Classification algorithms
    - Databricks for big data processing
    - Machine learning libraries

- **Approach/Methodology**:
    1. Data Analysis: Studied historical product launches
    2. Feature Engineering: Created relevant predictors
    3. Model Development: Built classification models
    4. Validation: Tested model accuracy
    5. Implementation: Deployed in Databricks

- **Outcome/Results**:
    - 40% improvement in launch success rates
    - 35% reduction in marketing costs
    - Better channel selection
    - More efficient resource allocation

- **Challenges & Solutions**:
    - Data Volume: Used Databricks for scalable processing
    - Model Selection: Implemented multiple algorithms
    - Integration: Created automated workflows

- **Impact**:
    The predictive model transformed product launch strategies, leading to more successful launches and better resource utilization.`
    },
    office4: {
        img: 'assets/project-office4.png',
        title: 'Proof Of Concept : AI Agent for Content Inventory Exploration',
        desc: 'Developed a proof-of-concept AI agent to help the Content Development Team explore and analyze content inventory efficiently.',
        tags: ['AI Agent', 'Content Team'],
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
        tags: ['AI Agent', 'Ops Team'],
        details: `
#### **POC 2: AI Agent for Reporting to Client (User: Operations Team)**

- **Objective**:
    - Developed an AI agent to automate and streamline the client reporting process, reducing manual effort and improving accuracy in report generation.
    - Aimed to standardize reporting formats and ensure consistent delivery of insights across different client accounts.

- **Key Technologies & Tools Used**:
    - Large Language Models (Claude 3.7 Sonnet) for natural language processing and report generation
    - Langchain framework for building the AI agent
    - Streamlit for creating an interactive user interface
    - Azure SQL Database for data storage and retrieval
    - Python libraries for data processing and visualization
    - Automated report generation tools and templates

- **Approach/Methodology**:
    1. Requirements Analysis: Identified key reporting needs and pain points in the current manual process
    2. Data Integration: Connected to various data sources and standardized data formats
    3. Agent Development: Built an AI agent capable of:
        - Understanding reporting requirements
        - Extracting relevant data
        - Generating formatted reports
        - Providing insights and recommendations
    4. UI Development: Created an intuitive interface for report customization and generation
    5. Testing & Validation: Conducted extensive testing with real client data and scenarios

- **Outcome/Results**:
    - Reduced report generation time by 70%
    - Improved accuracy in data analysis and reporting
    - Standardized reporting formats across all client accounts
    - Enhanced ability to handle multiple report types and formats
    - Increased client satisfaction with timely and consistent reports

- **Challenges & Solutions**:
    - Data Integration: Complex data sources with varying formats
        * Solution: Implemented robust data connectors and standardization processes
    - Report Customization: Different clients required different report formats
        * Solution: Created flexible templates and customization options
    - Accuracy Verification: Ensuring AI-generated reports were accurate
        * Solution: Implemented validation checks and human review processes

- **Impact**:
    The AI agent significantly transformed the operations team's reporting workflow. It eliminated manual data gathering and report creation, allowing team members to focus on analysis and client communication. The standardized reporting process improved consistency and reduced errors, leading to better client relationships and more efficient operations. The solution also provided a foundation for future automation initiatives in the organization.
    `
    },
    self1: {
        img: 'assets/project-self1.png',
        title: 'Market Mix Modeling',
        desc: 'Built a market mix model to analyze and optimize marketing spend across multiple channels for improved ROI.',
        tags: ['Python', 'Marketing Analytics'],
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
        tags: ['Python', 'Data Science'],
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
        tags: ['Python', 'RAG', 'NLP'],
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
        tags: ['Python', 'Classification'],
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
        tags: ['Python', 'EDA'],
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
