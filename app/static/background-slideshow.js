/*
 * Shared full-page background slideshow, used on every page (member-facing
 * and admin alike) for the cinematic CHAKAM look.
 *
 * Photo sets are split by section: the Photo Finder flow uses "outside"
 * photos, the Reaction Finder flow uses "inside" photos, and admin/utility
 * pages use a calmer "neutral" mix.
 */
const BG_PHOTOS = {
  outside: [
    "/static/images/IMG_0227.jpg",
    "/static/images/IMG_0279.jpg",
    "/static/images/IMG_2112.jpg",
    "/static/images/IMG_2198.jpg",
    "/static/images/IMG_5013.jpg",
    "/static/images/IMG_5728.jpg",
    "/static/images/IMG_5744.jpg",
    "/static/images/IMG_6992.jpg",
    "/static/images/IMG_9571.jpg",
    "/static/images/IMG_9575.jpg",
  ],
  inside: [
    "/static/images/IMG_5013.jpg",
    "/static/images/IMG_9571.jpg",
    "/static/images/IMG_9575.jpg",
    "/static/images/IMG_2112.jpg",
    "/static/images/IMG_0279.jpg",
  ],
  neutral: [
    "/static/images/IMG_0227.jpg",
    "/static/images/IMG_5728.jpg",
    "/static/images/IMG_5744.jpg",
    "/static/images/IMG_6992.jpg",
    "/static/images/IMG_2198.jpg",
  ],
};

const BG_SLIDESHOW_INTERVAL_MS = 7000;

function initBackgroundSlideshow(setName) {
  const container = document.querySelector(".bg-slideshow");
  if (!container) return;

  const photos = BG_PHOTOS[setName] || BG_PHOTOS.neutral;
  if (!photos || photos.length === 0) return;

  photos.forEach((url, i) => {
    const layer = document.createElement("div");
    layer.className = "bg-slideshow-layer";
    layer.style.backgroundImage = `url('${url}')`;
    if (i === 0) layer.classList.add("is-active");
    container.appendChild(layer);
  });

  if (photos.length <= 1) return;

  let index = 0;
  setInterval(() => {
    const layers = container.querySelectorAll(".bg-slideshow-layer");
    layers[index].classList.remove("is-active");
    index = (index + 1) % layers.length;
    layers[index].classList.add("is-active");
  }, BG_SLIDESHOW_INTERVAL_MS);
}
