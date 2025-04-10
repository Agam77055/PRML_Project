import bot from '../assets/bot.svg';
import user from '../assets/user.svg';
import './style.css';

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const form = document.querySelector('form');
  const chatContainer = document.querySelector('#chat_container');
  const textarea = document.querySelector('textarea');
  const welcomeScreen = document.querySelector('.welcome-screen');
  const submitButton = document.querySelector('form button[type="submit"]');
  const newChatBtn = document.querySelector('#new-chat-btn');

  newChatBtn.addEventListener('click', () => {
    // Reload the entire page when the button is clicked
    window.location.reload();
  });

  let loadInterval;
  let chatHistory = [];
  let currentConversationId = generateUniqueId();
  let isTyping = false; // Flag to track if the AI is currently typing

  // Initialize the chat interface
  function initializeUI() {
    welcomeScreen.style.display = 'flex';
    
    // Focus textarea
    setTimeout(() => {
      textarea.focus();
    }, 100);
  }

  // Auto resize textarea as user types
  textarea.addEventListener('input', () => {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
    
    // Enable/disable submit button based on input
    submitButton.disabled = textarea.value.trim() === '' || isTyping;
  });

  // Loading animation
  function loader(element) {
    element.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
  }

  // Format text with markdown-like syntax
  function formatText(text) {
    // Replace **bold** with <strong>bold</strong>
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Replace *italic* with <em>italic</em>
    text = text.replace(/\*((?!\*\*)[^*]+)\*/g, '<em>$1</em>');
    
    // Replace |quotation text with <blockquote>quotation text</blockquote>
    text = text.replace(/\|(.*?)(?:\n|$)/g, '<blockquote>$1</blockquote>');
    
    return text;
  }

  // Typing animation with formatted text support
  function typeText(element, text) {
    isTyping = true; // Set typing flag to true
    disableInput(); // Disable the input while typing
    
    // Format the text
    const formattedText = formatText(text);
    
    // Create a temporary div to parse the HTML
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = formattedText;
    const textContent = tempDiv.textContent; // Get the plain text content
    
    element.innerHTML = '';
    let index = 0;
    let interval = setInterval(() => {
      if (index < textContent.length) {
        // This approach ensures we're typing character by character but preserving formatting
        // We're recreating the HTML each time by formatting the appropriate substring of the original text
        const currentPosition = getPositionInOriginalText(text, textContent, index);
        element.innerHTML = formatText(text.substring(0, currentPosition + 1));
        index++;
        
        // Scroll to bottom as text is being typed
        chatContainer.scrollTop = chatContainer.scrollHeight;
      } else {
        clearInterval(interval);
        
        // Make sure the final text has all formatting applied
        element.innerHTML = formattedText;
        
        // Add timestamp after message is fully typed
        const timestamp = document.createElement('div');
        timestamp.className = 'timestamp';
        timestamp.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        element.parentNode.appendChild(timestamp);
        
        isTyping = false; // Reset typing flag
        enableInput(); // Re-enable the input after typing is complete
      }
    }, 20);
  }
  
  // Helper function to find the position in the original text that corresponds to a position in the plain text
  function getPositionInOriginalText(originalText, plainText, plainTextPosition) {
    // For simple cases, this might work adequately
    const plainSubstring = plainText.substring(0, plainTextPosition + 1);
    let originalPosition = 0;
    let plainPosition = 0;
    
    while (plainPosition < plainSubstring.length && originalPosition < originalText.length) {
      // Skip markdown characters in original text
      if ((originalText[originalPosition] === '*' && 
           (originalPosition + 1 < originalText.length && originalText[originalPosition + 1] === '*')) ||
          (originalText[originalPosition] === '*') ||
          (originalText[originalPosition] === '|')) {
        originalPosition++;
        continue;
      }
      
      if (originalText[originalPosition] === plainText[plainPosition]) {
        plainPosition++;
      }
      
      originalPosition++;
    }
    
    return originalPosition - 1;
  }

  // Disable input while AI is responding
  function disableInput() {
    textarea.disabled = true;
    submitButton.disabled = true;
  }

  // Enable input after AI response is complete
  function enableInput() {
    textarea.disabled = false;
    submitButton.disabled = textarea.value.trim() === '';
    textarea.focus();
  }

  // Generate unique ID for messages
  function generateUniqueId() {
    const timestamp = Date.now();
    const randomNumber = Math.random();
    const hexadecimalString = randomNumber.toString(16);
    return `id-${timestamp}-${hexadecimalString}`;
  }

  // Create HTML for chat messages
  function chatStripe(isAi, value, uniqueId) {
    return `
      <div class="message-wrapper ${isAi ? 'ai' : 'user'}">
        <div class="chat">
          <div class="profile">
            <img 
              src="${isAi ? bot : user}" 
              alt="${isAi ? 'bot' : 'user'}"
            />
          </div>
          <div class="message-container">
            <div class="message" id="${uniqueId}">${value}</div>
          </div>
        </div>
      </div>
    `;
  }

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // If AI is currently typing, don't allow submission
    if (isTyping) return;
    
    // Hide welcome screen when user sends first message
    if (welcomeScreen.style.display !== 'none') {
      welcomeScreen.style.display = 'none';
    }
    
    const data = new FormData(form);
    const userMessage = data.get('prompt').trim();
    
    if (!userMessage) return;
    
    // Reset textarea height
    textarea.style.height = 'auto';
    
    // Add user message to chat history
    const userMessageId = generateUniqueId();
    chatHistory.push({ id: userMessageId, isAi: false, value: userMessage });
    
    // User's chat stripe
    chatContainer.innerHTML += chatStripe(false, userMessage, userMessageId);
    form.reset();
    
    // Scroll to bottom
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    // Bot's chat stripe
    const botMessageId = generateUniqueId();
    chatContainer.innerHTML += chatStripe(true, "", botMessageId);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    const messageDiv = document.getElementById(botMessageId);
    loader(messageDiv);
    
    // Disable input while waiting for response
    disableInput();
    
    try {
      const response = await fetch('http://localhost:5001', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          prompt: userMessage,
          conversationId: currentConversationId
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        const parsedData = data.bot.trim();
                
        // Clear loading indicator
        clearInterval(loadInterval);
        messageDiv.innerHTML = '';
        
        // Type out the response with formatting
        typeText(messageDiv, parsedData);
        
        // Add bot message to chat history
        chatHistory.push({ id: botMessageId, isAi: true, value: parsedData });
      } else {
        // Handle server error
        messageDiv.innerHTML = '';
        const errorMessage = 'Something went wrong. Please try again.';
        messageDiv.innerHTML = `<div class="error">${errorMessage}</div>`;
        
        // Add error to chat history
        chatHistory.push({ id: botMessageId, isAi: true, value: errorMessage, error: true });
        
        isTyping = false;
        enableInput();
        
        const err = await response.text();
        console.error(err);
      }
    } catch (error) {
      // Handle network error
      messageDiv.innerHTML = '';
      const errorMessage = 'Network error. Please check your connection.';
      messageDiv.innerHTML = `<div class="error">${errorMessage}</div>`;
      
      // Add error to chat history
      chatHistory.push({ id: botMessageId, isAi: true, value: errorMessage, error: true });
      
      isTyping = false;
      enableInput();
      
      console.error(error);
    }
  };

  // Event listeners
  form.addEventListener('submit', handleSubmit);
  
  // Handle Enter key for submit (but Shift+Enter for new line)
  textarea.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (textarea.value.trim() !== '' && !isTyping) {
        handleSubmit(e);
      }
    }
  });
  
  // Initialize the UI
  initializeUI();
  
  // Add a welcome message when chat loads (if no chat history)
  if (chatHistory.length === 0) {
    setTimeout(() => {
      const welcomeId = generateUniqueId();
      const welcomeMessage = "Hi there! I'm your AI assistant. How can I help you today?";
      
      chatContainer.innerHTML += chatStripe(true, "", welcomeId);
      const messageDivWelcome = document.getElementById(welcomeId);
      
      setTimeout(() => {
        typeText(messageDivWelcome, welcomeMessage);
        
        // Add welcome message to chat history
        chatHistory.push({ id: welcomeId, isAi: true, value: welcomeMessage });
        
        // Hide welcome screen
        welcomeScreen.style.display = 'none';
      }, 500);
    }, 1000);
  }
});