function go_epochs(epochs_count) {
    fetch("http://localhost:5001/go_epochs", {
        method: "POST",
        headers: {
            "Content-Type": "application/x-www-form-urlencoded",
        },
        body: `epochs_count=${epochs_count}`,
        mode: "cors",
        credentials: "omit",
    })
        .then(response => response.json())
        .then(result => alert(result.message || result.error))
        .catch(error => alert("Error: " + error.message));
}

async function makePrediction(url) {
    if (!url || url.trim() === "") {
        alert("Please enter a valid URL");
        return;
    }

    try {
        const response = await fetch(`http://localhost:5001/predict?url=${encodeURIComponent(url)}`);
        var data = await response.json();
        if (response.ok) {
            data = JSON.parse(data);
            console.log(data);
            console.log(typeof data);

            const answer = data.predicted_number;
            const propabilities = data.prediction;

            const resultText = `
                <strong>Predicted number:</strong> ${answer}<br>
                <strong>All probabilities:</strong> [${propabilities.join(', ')}]
                <img src='${url}' width='20%' height='auto'>
            `;
            document.getElementById('test_result').innerHTML = resultText;
        } else {
            console.error("Error:", data.error);
            alert(`Error: ${data.error}`);
        }
    } catch (error) {
        console.error("Fetch error:", error);
        alert("Failed to fetch image");
    }
}

document.getElementById('test_button').addEventListener('click', () => {
    const urlInput = document.getElementById('url_test_image');
    makePrediction(urlInput.value);
});

async function ShowModelSize() {
    try {
        const response = await fetch(`http://localhost:5001/model_size`);
        var data = await response.json();
        if (response.ok) {
            data = JSON.parse(data);
            console.log(data);
            console.log(typeof data);

            const answer = data.size;
            document.getElementById('model_size').innerText = answer == 'None' ? 'Unknown' : answer;
        } else {
            console.error("Error:", data.error);
            alert(`Error: ${data.error}`);
        }
    } catch (error) {
        document.getElementById('model_size').innerText = "Couldnt define model size: " + error;
        console.error("Fetch error:", error);
        alert("Failed to fetch image");
    }
}

ShowModelSize();