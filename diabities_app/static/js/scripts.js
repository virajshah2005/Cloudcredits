document.addEventListener('DOMContentLoaded', () => {
    // Smooth scroll for nav links
    const navLinks = document.querySelectorAll('.nav-links a');

    navLinks.forEach(link => {
        link.addEventListener('click', e => {
            if (link.hash !== '') {
                e.preventDefault();
                const target = document.querySelector(link.hash);
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth' });
                }
            }
        });
    });

    // Form validation for diabetes prediction form
    const form = document.querySelector('form[action="/predict"]');
    if (form) {
        form.addEventListener('submit', (e) => {
            const inputs = form.querySelectorAll('input[type="number"]');
            let allFilled = true;
            inputs.forEach(input => {
                if (!input.value) {
                    allFilled = false;
                    input.classList.add('input-error');
                } else {
                    input.classList.remove('input-error');
                }
            });
            if (!allFilled) {
                e.preventDefault();
                alert('Please fill in all the fields before submitting the form.');
            }
        });
    }
});
