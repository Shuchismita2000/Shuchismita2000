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

// Project modal logic
const projectData = {
    office1: {
        img: 'project-office1.jpg',
        title: 'Customer Risk Prediction',
        desc: 'Developed and deployed ML models to Databricks endpoints for real-time customer risk prediction, improving underwriting efficiency.',
        tags: ['Python', 'Databricks', 'MLflow']
    },
    office2: {
        img: 'project-office2.jpg',
        title: 'Market-Mix Modeling',
        desc: 'Prepared end-to-end solutions in Azure ML Studio for market-mix modeling and dynamic pricing for price optimization.',
        tags: ['Azure ML Studio', 'Regression']
    },
    self1: {
        img: 'project-self1.jpg',
        title: 'Traffic Sign Classification',
        desc: 'Designed and implemented a deep learning pipeline using CNNs to classify traffic signs from image datasets.',
        tags: ['Python', 'Deep Learning', 'CNN']
    },
    self2: {
        img: 'project-self2.jpg',
        title: 'Uplift Modeling for Marketing',
        desc: 'Implemented uplift modeling techniques to predict the incremental impact of marketing strategies for personalized campaigns.',
        tags: ['Python', 'Uplift Modeling']
    }
};

const modal = document.getElementById('project-modal');
const modalImg = document.querySelector('.modal-img');
const modalTitle = document.querySelector('.modal-title');
const modalDesc = document.querySelector('.modal-desc');
const modalTags = document.querySelector('.modal-tags');
const modalClose = document.querySelector('.modal-close');

// Open modal on card click
Array.from(document.querySelectorAll('.project-card')).forEach(card => {
    card.addEventListener('click', () => {
        const key = card.getAttribute('data-project');
        const data = projectData[key];
        if (data) {
            modalImg.src = data.img;
            modalImg.alt = data.title;
            modalTitle.textContent = data.title;
            modalDesc.textContent = data.desc;
            modalTags.innerHTML = data.tags.map(tag => `<span>${tag}</span>`).join(' ');
            modal.classList.add('active');
        }
    });
});

// Close modal on close button or overlay click
modalClose.addEventListener('click', () => modal.classList.remove('active'));
modal.addEventListener('click', e => {
    if (e.target === modal) modal.classList.remove('active');
}); 