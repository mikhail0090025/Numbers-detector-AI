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