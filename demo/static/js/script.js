document.getElementById('question-form').addEventListener('submit', function(e) {
    e.preventDefault();
    const question = document.getElementById('question').value;
    const output = document.getElementById('output');
    output.innerHTML = ''; // Clear previous output

    const formData = new FormData();
    formData.append('question', question);

    const eventSource = new EventSource('/generate?' + new URLSearchParams(formData));

    let fullResponse = '';

    eventSource.onmessage = function(event) {
        if (event.data === '[DONE]') {
            eventSource.close();
            renderStructuredOutput(fullResponse);
        } else {
            fullResponse += event.data;
            output.textContent = fullResponse; // Display raw text during streaming
        }
    };

    eventSource.onerror = function() {
        output.textContent = 'Error occurred during generation.';
        eventSource.close();
    };
});

function renderStructuredOutput(response) {
    const output = document.getElementById('output');
    output.innerHTML = ''; // Clear raw text

    // Parse the response to extract sections
    const sections = parseResponse(response);
    
    sections.forEach(section => {
        const div = document.createElement('div');
        div.className = section.type;
        const title = document.createElement('strong');
        title.textContent = section.type.charAt(0).toUpperCase() + section.type.slice(1) + ':';
        div.appendChild(title);
        
        const content = document.createElement('pre');
        content.textContent = section.content;
        div.appendChild(content);
        
        output.appendChild(div);
    });
}

function parseResponse(response) {
    const sections = [];
    const regex = /<(\w+)>(.*?)<\/\1>/gs;
    let match;
    let lastIndex = 0;

    while ((match = regex.exec(response)) !== null) {
        const type = match[1].toLowerCase();
        const content = match[2].trim();
        
        // Add any content before this tag as 'extra'
        if (match.index > lastIndex) {
            const extraContent = response.slice(lastIndex, match.index).trim();
            if (extraContent) {
                sections.push({ type: 'extra', content: extraContent });
            }
        }
        
        sections.push({ type, content });
        lastIndex = regex.lastIndex;
    }

    // Add any remaining content after the last tag
    if (lastIndex < response.length) {
        const extraContent = response.slice(lastIndex).trim();
        if (extraContent) {
            sections.push({ type: 'extra', content: extraContent });
        }
    }

    return sections;
}