document.addEventListener('DOMContentLoaded', function () {
    var content = document.querySelector('.content');
    var isContentVisible = false;

    function isElementInViewport(el) {
        var rect = el.getBoundingClientRect();
        return (
            rect.top >= 0 &&
            rect.left >= 0 &&
            rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
            rect.right <= (window.innerWidth || document.documentElement.clientWidth)
        );
    }

    function handleScroll() {
        var isVisible = isElementInViewport(content);

        if (isVisible && !isContentVisible) {
            content.classList.add('show');
            isContentVisible = true;

            // Remove scroll event listener after the animation has played
            window.removeEventListener('scroll', handleScroll);
        }
    }

    window.addEventListener('scroll', handleScroll);
    handleScroll(); // Check on initial load
});

document.querySelector('.container label.footer').addEventListener('click', function () {
    document.getElementById('file').click();
});

document.getElementById('file').addEventListener('change', function () {
    var fileName = this.value.split('\\').pop();
    document.querySelector('.container label.footer p').innerText = fileName || 'Not selected file';
});

// Attach event listener directly to the form's submit event
document.getElementById('predict').addEventListener('submit', function(event) {
    // Prevent the default form submission behavior
    event.preventDefault();
    
    // Clear previous error messages
    document.getElementById('error-message').innerHTML = '';

    // Validate file format
    var fileInput = document.getElementById('file');
    var fileName = fileInput.value;
    if (!fileName.endsWith('.csv')) {
        showError('Error: Dataset must be in .csv format.');
        return;
    }

    // Submit the form if validation passes
    this.submit();
});

function showError(message) {
    var errorMessage = document.getElementById('error-message');
    errorMessage.innerText = message;
    errorMessage.style.display = 'block';
}