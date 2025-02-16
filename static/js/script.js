document.getElementById('checkButton').addEventListener('click', async () => {
    const text = document.getElementById('inputText').value;
    const resultsDiv = document.getElementById('results');

    if (!text.trim()) {
        resultsDiv.innerHTML = '<p>Please enter text to check.</p>';
        return;
    }

    resultsDiv.innerHTML = '<p>Checking...</p>';

    try {
        const response = await fetch('/check-plagiarism', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text }),
        });

        const result = await response.json();
        if (response.ok) {
            resultsDiv.innerHTML = `<p>Plagiarism Result: ${JSON.stringify(result)}</p>`;
        } else {
            resultsDiv.innerHTML = `<p>Error: ${result.error}</p>`;
        }
    } catch (error) {
        resultsDiv.innerHTML = `<p>Error: ${error.message}</p>`;
    }
});
