document.addEventListener('DOMContentLoaded', () => {
    const questionInput = document.getElementById('questionInput');
    const askButton = document.getElementById('askButton');
    const buttonText = document.getElementById('buttonText');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const responseDisplay = document.getElementById('responseDisplay');
    const clearPineconeButton = document.getElementById('clearPineconeButton');
    const clearMessage = document.getElementById('clearMessage');

    // Function to enable/disable UI elements during processing
    function setProcessingState(isProcessing) {
        askButton.disabled = isProcessing;
        questionInput.disabled = isProcessing;
        clearPineconeButton.disabled = isProcessing;
        if (isProcessing) {
            buttonText.textContent = 'Processing...';
            loadingSpinner.classList.remove('hidden');
            responseDisplay.textContent = 'Initiating data ingestion and generating response...';
        } else {
            buttonText.textContent = 'Get Answer';
            loadingSpinner.classList.add('hidden');
        }
    }

    // Function to handle asking a question with combined ingestion
    askButton.addEventListener('click', async () => {
        const question = questionInput.value.trim();
        if (!question) {
            responseDisplay.textContent = 'Please enter a question.';
            return;
        }

        setProcessingState(true);
        responseDisplay.textContent = ''; // Clear previous response

        try {
            const response = await fetch(`/ask_llm_with_ingestion/?question=${encodeURIComponent(question)}`);

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let result = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) {
                    break;
                }
                const chunk = decoder.decode(value, { stream: true });
                result += chunk;
                responseDisplay.textContent = result; // Update content as it streams
                responseDisplay.scrollTop = responseDisplay.scrollHeight; // Auto-scroll to bottom
            }

        } catch (error) {
            console.error('Error during combined ingestion and LLM query:', error);
            responseDisplay.textContent = `Error: ${error.message}. Please check server logs.`;
        } finally {
            setProcessingState(false);
        }
    });

    // Function to clear Pinecone index
    clearPineconeButton.addEventListener('click', async () => {
        if (!confirm('Are you sure you want to clear ALL data from the Pinecone index? This action cannot be undone.')) {
            return;
        }

        clearMessage.textContent = 'Clearing Pinecone index...';
        clearPineconeButton.disabled = true;

        try {
            const response = await fetch('/clear_pinecone_index/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            const data = await response.json();
            if (response.ok) {
                clearMessage.textContent = data.message;
                console.log(data.message);
            } else {
                clearMessage.textContent = `Error: ${data.detail || 'Failed to clear index.'}`;
                console.error('Error clearing Pinecone index:', data.detail);
            }
        } catch (error) {
            clearMessage.textContent = `Network error: ${error.message}`;
            console.error('Network error clearing Pinecone index:', error);
        } finally {
            clearPineconeButton.disabled = false;
        }
    });
});