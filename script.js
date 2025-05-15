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
        img: 'project-office4.jpg',
        title: 'POC: AI Agent for Content Inventory Exploration',
        desc: 'Developed a proof-of-concept AI agent to help the Content Development Team explore and analyze content inventory efficiently.',
        tags: ['AI Agent', 'Content Team']
    },
    office5: {
        img: 'project-office5.jpg',
        title: 'POC: AI Agent for Reporting to Client',
        desc: 'Created a proof-of-concept AI agent for the Ops Team to automate and streamline client reporting processes.',
        tags: ['AI Agent', 'Ops Team']
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

const modal = document.getElementById('project-modal');
const modalImg = document.querySelector('.modal-img');
const modalTitle = document.querySelector('.modal-title');
const modalDesc = document.querySelector('.modal-desc');
const modalTags = document.querySelector('.modal-tags');
const modalClose = document.querySelector('.modal-close');

// Open modal on card click with safety check
Array.from(document.querySelectorAll('.project-card')).forEach(card => {
    card.addEventListener('click', () => {
        const key = card.getAttribute('data-project');
        const data = projectData[key];
        if (!data) {
            alert('Sorry, project details not available yet!');
            return;
        }
        modalImg.src = data.img;
        modalImg.alt = data.title;
        modalTitle.textContent = data.title;
        modalDesc.textContent = data.desc;
        modalTags.innerHTML = data.tags.map(tag => `<span>${tag}</span>`).join(' ');
        modal.classList.add('active');
    });
});

// Close modal on close button or overlay click
modalClose.addEventListener('click', () => modal.classList.remove('active'));
modal.addEventListener('click', e => {
    if (e.target === modal) modal.classList.remove('active');
});
