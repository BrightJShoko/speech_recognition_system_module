const recordButton = document.getElementById("recordButton");
const stopButton = document.getElementById("stopButton");
const statusDisplay = document.getElementById("status");

let mediaRecorder;
let audioChunks = [];

recordButton.addEventListener("click", async () => {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream);

  mediaRecorder.ondataavailable = (event) => {
    audioChunks.push(event.data);
  };

  mediaRecorder.onstop = async () => {
    const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
    const formData = new FormData();
    formData.append("audio_data", audioBlob, "recording.wav");

    try {
      const response = await fetch("/upload", {
        method: "POST",
        body: formData,
      });
      if (response.ok) {
        statusDisplay.textContent = "Audio uploaded successfully!";
      } else {
        statusDisplay.textContent = "Failed to upload audio.";
      }
    } catch (error) {
      statusDisplay.textContent = "Error uploading audio.";
      console.error("Error:", error);
    }

    audioChunks = [];
  };

  mediaRecorder.start();
  recordButton.disabled = true;
  stopButton.disabled = false;
  statusDisplay.textContent = "Recording...";
});

stopButton.addEventListener("click", () => {
  mediaRecorder.stop();
  recordButton.disabled = false;
  stopButton.disabled = true;
  statusDisplay.textContent = "Recording stopped. Uploading...";
});
