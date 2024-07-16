// custom.js
// Add button to expand images to full screen
document.addEventListener("DOMContentLoaded", function() {
    const images = document.querySelectorAll("img:not(.no-expand)");
	const theme = localStorage.getItem("theme") || "light";
	document.documentElement.setAttribute('data-theme', theme);

    images.forEach((img) => {
		img.onload = function() {
			if (img.naturalWidth > 100 && img.naturalHeight > 100) {
				const button = document.createElement("button");
				button.innerHTML = "&#x2B0D;"; // Unicode for the square with diagonal arrows

				button.style.position = "absolute";
				button.style.top = "10px";
				button.style.right = "10px";
				button.style.width = "40px"; // Adjust size as needed
				button.style.height = "40px"; // Adjust size as needed
				button.style.fontSize = "20px"; // Adjust size as needed
				button.style.background = "rgba(0, 0, 0, 0.7)";
				button.style.color = "white";
				button.style.border = "none";
				button.style.cursor = "pointer";
				button.style.borderRadius = "50%";
				button.style.boxShadow = "0 2px 4px rgba(0, 0, 0, 0.2)";
				button.style.transition = "background-color 0.3s, transform 0.3s";

				const wrapper = document.createElement("div");
				wrapper.style.position = "relative";
				wrapper.style.display = "inline-block";
				wrapper.style.backgroundColor = "white"; // Set background color of the wrapper div
				img.style.backgroundColor = "white"; // Set background color of the img element

				img.parentNode.insertBefore(wrapper, img);
				wrapper.appendChild(img);
				wrapper.appendChild(button);

				button.addEventListener("click", function() {
					if (document.fullscreenElement) {
						document.exitFullscreen();
						button.innerHTML = "&#x2B0D;"; // Reset icon on exit
					} else {
						img.requestFullscreen();
						//button.innerHTML = "&#x274C;"; // Change to close icon when in fullscreen
					}
				});

				// Style change on hover
				button.addEventListener("mouseenter", function() {
					button.style.backgroundColor = "rgba(0, 0, 0, 0.9)";
				});

				button.addEventListener("mouseleave", function() {
					button.style.backgroundColor = "rgba(0, 0, 0, 0.7)";
				});
			}
		};
    });
});
