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
        img: 'project-office1.jpg',
        title: 'Market Mix Modeling for Optimized Budget Allocation',
        desc: 'Prepared end-to-end solutions in Azure ML Studio for market-mix modeling and dynamic pricing for price optimization.',
        tags: ['Azure ML Studio', 'Regression']
    },
    office2: {
        img: 'project-office2.jpg',
        title: 'Dynamic Pricing for Optimizing Pricing Strategy',
        desc: 'Implemented dynamic pricing models to optimize product pricing strategies, increasing revenue and market competitiveness.',
        tags: ['Azure ML Studio', 'Regression']
    },
    office3: {
        img: 'project-office3.jpg',
        title: 'Predicting Optimal Channel for New Product Launch',
        desc: 'Built predictive models to identify the most effective marketing channels for launching new products, improving launch success rates.',
        tags: ['Python', 'Classification', 'Databricks']
    },
    office4: {
        img: 'assets/project-office4.png',
        title: 'POC: AI Agent for Content Inventory Exploration',
        desc: 'Developed a proof-of-concept AI agent to help the Content Development Team explore and analyze content inventory efficiently.',
        tags: ['AI Agent', 'Content Team'],
        details: `
- **Objective**:
    
    The primary goal of creating this AI agent was to automate and simplify the exploration and management of the extensive content inventory maintained by the Enqurious content team. With hundreds of content pieces produced regularly, it became difficult to keep track of published items, especially those needing fixes or ready to be showcased to clients. The agent was designed to reduce manual effort, speed up content discovery, and support creating a clean catalog of deliverable content.
    
- **Key Technologies & Tools Used**:
    
    - AI and NLP: Utilized a large language model (Claude 3.7 Sonnet) for natural language understanding and response generation.
        
    - Langchain framework: Enabled building AI applications and agents with ease.
        
    - Streamlit: Provided a user-friendly chat-like interface for interaction with the agent.
        
    - Azure SQL Database: Source of the content data.
        
    - Google Sheets & CSV files: Used for data visibility and to simplify the proof-of-concept setup.
        
    - Python libraries (.env for secure key management, Langchain-anthropic for AI integration).
        
- **Approach/Methodology**:
    
    1. **Understanding the Problem**: Identified the pain points of manual content inventory checks and inconsistent tracking.
        
    2. **Data Preparation**: Data was extracted from Azure SQL to Google Sheets and downloaded as CSV files. Data cleaning involved fixing inconsistent column names, formatting, and removing errors.
        
    3. **Organizing Data**: Content was partitioned into logical folders by creator and content status (draft, published, archived) to simplify data access.
        
    4. **Building the Agent**: Developed a CSV agent using Langchain connected to the cleaned CSV files. The agent could perform SQL-like queries based on user text prompts.
        
    5. **Creating a UI**: Built a Streamlit app to simulate a chat interface for natural user interaction.
        
    6. **Prompt Engineering**: Designed prompts to specify content creator, status, and query type, ensuring relevant and formatted output.
        
    7. **Testing and Iteration**: Tested with real user queries, identified inconsistencies and performance issues, and outlined future improvements.
        
- **Outcome/Results**:
    
    - Significant reduction in manual effort needed to find, filter, and verify content status.
        
    - Faster content discovery and generation of reports in a consistent format.
        
    - Improved accuracy in identifying content ready for client delivery.
        
    - User-friendly interface enabled easy adoption by content developers without technical expertise.
        
    - Modular design prepared the system for scaling and integration with cloud infrastructure.
        
- **Challenges & Solutions**:
    
    - **Data Quality Issues**: Messy data with inconsistent formats and missing entries initially hindered performance. Addressed by thorough data cleaning and standardized formatting.
        
    - **Response Inconsistency**: AI agent sometimes returned answers in varying formats (tables, bullets, paragraphs), which complicated report generation. Future work aims to enforce unified output templates.
        
    - **Latency**: Using a free AI model led to slow response times (8–10+ seconds). Solution involves optimizing queries and considering paid, faster APIs.
        
    - **Lack of Performance Metrics**: No automated way to measure accuracy or user satisfaction. Planned integration of feedback loops and monitoring metrics.
        
- **Impact**:
    
    This AI agent fundamentally transformed how the content development team interacted with their content inventory. It provided an intelligent, automated assistant capable of swiftly answering questions and generating reports, thus saving time and reducing human error. By automating content exploration, the agent improved the quality and timeliness of client-ready content catalogs and empowered the team to focus more on content creation rather than inventory management. The approach laid a solid foundation for scaling content operations and integrating advanced AI-driven workflows in the future.`
    },
    office5: {
        img: 'project-office5.jpg',
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
        img: 'project-self1.jpg',
        title: 'Market Mix Modeling',
        desc: 'Built a market mix model to analyze and optimize marketing spend across multiple channels for improved ROI.',
        tags: ['Python', 'Marketing Analytics']
    },
    self2: {
        img: 'project-self2.jpg',
        title: 'Water Profiling Score Project',
        desc: 'Developed a scoring system to profile water quality using data analysis and machine learning techniques.',
        tags: ['Python', 'Data Science']
    },
    self3: {
        img: 'project-self3.jpg',
        title: 'Book Recommendation (RAG Chain)',
        desc: 'Implemented a book recommendation system using Retrieval-Augmented Generation (RAG) and NLP techniques.',
        tags: ['Python', 'RAG', 'NLP']
    },
    self4: {
        img: 'project-self4.jpg',
        title: 'Customer Churn Prediction',
        desc: 'Created predictive models to identify customers at risk of churning, enabling targeted retention strategies.',
        tags: ['Python', 'Classification']
    },
    self5: {
        img: 'project-self5.jpg',
        title: 'Customer Purchase Behavior',
        desc: 'Analyzed and modeled customer purchase patterns to uncover insights and drive business growth.',
        tags: ['Python', 'EDA']
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
