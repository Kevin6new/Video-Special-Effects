function applyFilter(filterName) {
    fetch(`http://localhost:8000/apply-${filterName}`)
        .then(response => {
            if (response.ok) {
                console.log(`${filterName} filter applied.`);
            } else {
                console.error(`Error: Server responded with status ${response.status}`);
            }
        })
        .catch(error => console.error('Error:', error));
}

document.getElementById('greyscale').addEventListener('click', function () {
    applyFilter('greyscale');
});

document.getElementById('sepiaTone').addEventListener('click', function () {
    applyFilter('sepia');
});

document.getElementById('blur5x5').addEventListener('click', function () {
    applyFilter('blur5x5');
});

document.getElementById('blur5x5Alt').addEventListener('click', function () {
    applyFilter('blur5x5-alt');
});

document.getElementById('sobelX').addEventListener('click', function () {
    applyFilter('sobel-x');
});

document.getElementById('sobelY').addEventListener('click', function () {
    applyFilter('sobel-y');
});

document.getElementById('magnitude').addEventListener('click', function () {
    applyFilter('magnitude');
});

document.getElementById('blurQuantize').addEventListener('click', function () {
    applyFilter('blur-quantize');
});

// Additional buttons
document.getElementById('detectFaces').addEventListener('click', function () {
    applyFilter('detect-faces');
});

document.getElementById('cartoon').addEventListener('click', function () {
    applyFilter('cartoon');
});

document.getElementById('negative').addEventListener('click', function () {
    applyFilter('negative');
});

document.getElementById('colorizeFaces').addEventListener('click', function () {
    applyFilter('colorize-faces');
});

document.getElementById('increaseBrightness').addEventListener('click', function () {
    applyFilter('increase-brightness');
});

document.getElementById('decreaseBrightness').addEventListener('click', function () {
    applyFilter('decrease-brightness');
});

document.getElementById('increaseContrast').addEventListener('click', function () {
    applyFilter('increase-contrast');
});

document.getElementById('decreaseContrast').addEventListener('click', function () {
    applyFilter('decrease-contrast');
});

document.getElementById('quitButton').addEventListener('click', function () {
    fetch('http://localhost:8000/quit')
        .then(response => {
            if (response.ok) {
                console.log("Quit command sent");
                
            }
        })
        .catch(error => console.error('Error sending quit command:', error));
});
