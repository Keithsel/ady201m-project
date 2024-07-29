document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('prediction-form');
    form.addEventListener('submit', function(event) {
        event.preventDefault();
        const formData = new FormData(form);
        const data = {
            automaker: formData.get('automaker'),
            location: formData.get('location'),
            year: formData.get('year'),
            kilometers_driven: formData.get('kilometers_driven'),
            fuel_type: formData.get('fuel_type'),
            transmission: formData.get('transmission'),
            owner_type: formData.get('owner_type'),
            mileage: formData.get('mileage'),
            engine: formData.get('engine'),
            power: formData.get('power'),
            seats: formData.get('seats')
        };
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(result => {
            const resultDiv = document.getElementById('prediction-result');
            resultDiv.textContent = `Predicted Price: ${result.prediction}`;
        });
    });
});
