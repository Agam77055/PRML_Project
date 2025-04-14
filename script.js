let currentIndex = 0;
let carouselInterval;
let totalImages;

function updateCarousel(animate = true) {
  const carouselImages = document.querySelector('.carousel-images');
  if (animate) {
    carouselImages.style.transition = 'transform 0.5s ease';
  } else {
    carouselImages.style.transition = 'none';
  }
  const offset = -currentIndex * 100;
  carouselImages.style.transform = `translateX(${offset}%)`;
}

function prevImage() {
  if (currentIndex === 0) {
    currentIndex = totalImages - 1;
    updateCarousel(false); // jump to last real image instantly
    requestAnimationFrame(() => {
      currentIndex--;
      updateCarousel();
    });
  } else {
    currentIndex--;
    updateCarousel();
  }
  resetTimer();
}

function nextImage() {
  currentIndex++;
  updateCarousel();

  if (currentIndex === totalImages) {
    setTimeout(() => {
      currentIndex = 0;
      updateCarousel(false); // reset without animation
    }, 500); // match animation duration
  }
  resetTimer();
}

function startCarouselTimer() {
  carouselInterval = setInterval(() => {
    nextImage();
  }, 3000); // change image every 3 seconds
}

function resetTimer() {
  clearInterval(carouselInterval);
  startCarouselTimer();
}

document.addEventListener("DOMContentLoaded", () => {
  const carouselImages = document.querySelector('.carousel-images');
  const images = document.querySelectorAll('.carousel-image');
  totalImages = images.length;

  // Clone the first image and append to the end
  const firstClone = images[0].cloneNode(true);
  carouselImages.appendChild(firstClone);

  updateCarousel(false);
  startCarouselTimer();

  // Pause on hover
  const carousel = document.querySelector('.carousel');
  carousel.addEventListener('mouseenter', () => clearInterval(carouselInterval));
  carousel.addEventListener('mouseleave', startCarouselTimer);
});
