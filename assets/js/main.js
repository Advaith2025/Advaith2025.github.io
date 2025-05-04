/**
* Template Name: iPortfolio
* Template URL: https://bootstrapmade.com/iportfolio-bootstrap-portfolio-websites-template/
* Updated: Jun 29 2024 with Bootstrap v5.3.3
* Author: BootstrapMade.com
* License: https://bootstrapmade.com/license/
*/

(function() {
  "use strict";

  /**
   * Header toggle
   */
  const headerToggleBtn = document.querySelector('.header-toggle');

  function headerToggle() {
    document.querySelector('#header').classList.toggle('header-show');
    headerToggleBtn.classList.toggle('bi-list');
    headerToggleBtn.classList.toggle('bi-x');
  }
  headerToggleBtn.addEventListener('click', headerToggle);

  /**
   * Hide mobile nav on same-page/hash links
   */
  document.querySelectorAll('#navmenu a').forEach(navmenu => {
    navmenu.addEventListener('click', () => {
      if (document.querySelector('.header-show')) {
        headerToggle();
      }
    });

  });

  /**
   * Toggle mobile nav dropdowns
   */
  document.querySelectorAll('.navmenu .toggle-dropdown').forEach(navmenu => {
    navmenu.addEventListener('click', function(e) {
      e.preventDefault();
      this.parentNode.classList.toggle('active');
      this.parentNode.nextElementSibling.classList.toggle('dropdown-active');
      e.stopImmediatePropagation();
    });
  });

  /**
   * Preloader
   */
  const preloader = document.querySelector('#preloader');
  if (preloader) {
    window.addEventListener('load', () => {
      preloader.remove();
    });
  }

  /**
   * Scroll top button
   */
  let scrollTop = document.querySelector('.scroll-top');

  function toggleScrollTop() {
    if (scrollTop) {
      window.scrollY > 100 ? scrollTop.classList.add('active') : scrollTop.classList.remove('active');
    }
  }
  scrollTop.addEventListener('click', (e) => {
    e.preventDefault();
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });
  });

  window.addEventListener('load', toggleScrollTop);
  document.addEventListener('scroll', toggleScrollTop);

  /**
   * Animation on scroll function and init
   */
  function aosInit() {
    AOS.init({
      duration: 600,
      easing: 'ease-in-out',
      once: true,
      mirror: false
    });
  }
  window.addEventListener('load', aosInit);

  /**
   * Init typed.js
   */
  const selectTyped = document.querySelector('.typed');
  if (selectTyped) {
    let typed_strings = selectTyped.getAttribute('data-typed-items');
    typed_strings = typed_strings.split(',');
    new Typed('.typed', {
      strings: typed_strings,
      loop: true,
      typeSpeed: 100,
      backSpeed: 50,
      backDelay: 2000
    });
  }

  /**
   * Initiate Pure Counter
   */
  new PureCounter();

  /**
   * Animate the skills items on reveal
   */
  let skillsAnimation = document.querySelectorAll('.skills-animation');
  skillsAnimation.forEach((item) => {
    new Waypoint({
      element: item,
      offset: '80%',
      handler: function(direction) {
        let progress = item.querySelectorAll('.progress .progress-bar');
        progress.forEach(el => {
          el.style.width = el.getAttribute('aria-valuenow') + '%';
        });
      }
    });
  });

  /**
   * Initiate glightbox
   */
  const glightbox = GLightbox({
    selector: '.glightbox'
  });

  /**
   * Init isotope layout and filters
   */
  /* // Commenting out Isotope again for custom layout
  document.querySelectorAll('.isotope-layout').forEach(function(isotopeItem) {
    let layout = isotopeItem.getAttribute('data-layout') ?? 'masonry';
    let filter = isotopeItem.getAttribute('data-default-filter') ?? '*';
    let sort = isotopeItem.getAttribute('data-sort') ?? 'original-order';

    let initIsotope;
    imagesLoaded(isotopeItem.querySelector('.isotope-container'), function() {
      initIsotope = new Isotope(isotopeItem.querySelector('.isotope-container'), {
        itemSelector: '.isotope-item',
        layoutMode: layout, // Ensure layout is masonry (or chosen mode)
        filter: filter,
        sortBy: sort
      });
    });

    isotopeItem.querySelectorAll('.isotope-filters li').forEach(function(filters) {
      filters.addEventListener('click', function() {
        isotopeItem.querySelector('.isotope-filters .filter-active').classList.remove('filter-active');
        this.classList.add('filter-active');
        initIsotope.arrange({
          filter: this.getAttribute('data-filter')
        });
        if (typeof aosInit === 'function') {
          aosInit();
        }
      }, false);
    });

  });
  */ // End commented Isotope block 

  /**
   * Init swiper sliders
   */
  function initSwiper() {
    document.querySelectorAll(".init-swiper").forEach(function(swiperElement) {
      let config = JSON.parse(
        swiperElement.querySelector(".swiper-config").innerHTML.trim()
      );

      if (swiperElement.classList.contains("swiper-tab")) {
        initSwiperWithCustomPagination(swiperElement, config);
      } else {
        new Swiper(swiperElement, config);
      }
    });
  }

  window.addEventListener("load", initSwiper);

  /** 
   * Init Portfolio Swiper 
   */
  /* // Commenting out Swiper initialization
  const portfolioSwiper = new Swiper('.portfolio-swiper', {
    // Configuration changes
    loop: false,          // Disable looping
    slidesPerView: 1,     // Show only one slide
    spaceBetween: 30,   // Adjust space if needed
    // autoHeight: true,     // Temporarily disable for debugging
  
    // Navigation arrows
    navigation: {
      nextEl: '.swiper-button-next',
      prevEl: '.swiper-button-prev',
    },

    // Optional: Pagination
    // pagination: {
    //   el: '.swiper-pagination',
    //   clickable: true,
    // },

    // Make slides clickable (useful if linking the whole slide)
    // slideToClickedSlide: true,

    // Responsive breakpoints
    // breakpoints: {
    //   // when window width is >= 640px
    //   640: {
    //     slidesPerView: 2,
    //     spaceBetween: 20
    //   },
    //   // when window width is >= 992px
    //   992: {
    //     slidesPerView: 3,
    //     spaceBetween: 30
    //   }
    // }
  });
  */

  /**
   * Correct scrolling position upon page load for URLs containing hash links.
   */
  window.addEventListener('load', function(e) {
    if (window.location.hash) {
      if (document.querySelector(window.location.hash)) {
        setTimeout(() => {
          let section = document.querySelector(window.location.hash);
          let scrollMarginTop = getComputedStyle(section).scrollMarginTop;
          window.scrollTo({
            top: section.offsetTop - parseInt(scrollMarginTop),
            behavior: 'smooth'
          });
        }, 100);
      }
    }
  });

  /**
   * Navmenu scrollspy
   */
  let navmenulinks = document.querySelectorAll('.navmenu a');

  function navmenuScrollspy() {
    navmenulinks.forEach(navmenulink => {
      if (!navmenulink.hash) return;
      let section = document.querySelector(navmenulink.hash);
      if (!section) return;
      let position = window.scrollY + 200;
      if (position >= section.offsetTop && position <= (section.offsetTop + section.offsetHeight)) {
        document.querySelectorAll('.navmenu a.active').forEach(link => link.classList.remove('active'));
        navmenulink.classList.add('active');
      } else {
        navmenulink.classList.remove('active');
      }
    })
  }
  window.addEventListener('load', navmenuScrollspy);
  document.addEventListener('scroll', navmenuScrollspy);

  /**
   * Digital Clock in Sidebar
   */
  const hoursElement = document.getElementById('clock-hours');
  const minutesElement = document.getElementById('clock-minutes');
  const secondsElement = document.getElementById('clock-seconds');
  const millisecondsElement = document.getElementById('clock-milliseconds'); // Get milliseconds element
  const dateElement = document.getElementById('clock-date');

  // Options for formatting the date (e.g., Fri Apr 25)
  const dateOptions = { weekday: 'short', month: 'short', day: 'numeric' };

  function updateDigitalClock() {
    const now = new Date();

    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    const seconds = String(now.getSeconds()).padStart(2, '0'); // Get whole seconds, padded
    const milliseconds = String(now.getMilliseconds()).padStart(3, '0'); // Format milliseconds to 3 digits
    
    // Format date like "Fri Apr 25"
    const formattedDate = now.toLocaleDateString('en-US', dateOptions).replace(/,/g, '');

    if (hoursElement) {
        hoursElement.textContent = hours;
    }
    if (minutesElement) {
        minutesElement.textContent = minutes;
    }
    if (secondsElement) {
        secondsElement.textContent = seconds; // Update with whole seconds (SS)
    }
    if (millisecondsElement) {
        millisecondsElement.textContent = milliseconds; // Update milliseconds
    }
    if (dateElement) {
      dateElement.textContent = formattedDate;
    }
  }

  // Update the clock frequently for milliseconds
  if (hoursElement && minutesElement && secondsElement && millisecondsElement && dateElement) {
      setInterval(updateDigitalClock, 50); // Update every 50ms
      updateDigitalClock(); // Initial call to display immediately
  }

  /**
   * Mouse Follower Effect (Orb with Inertia)
   */
  const orbElement = document.documentElement; // Use root element to set CSS vars
  let targetX = window.innerWidth / 2;
  let targetY = window.innerHeight / 2;
  let currentX = window.innerWidth / 2;
  let currentY = window.innerHeight / 2;
  const dampingFactor = 0.1; // Adjust for more/less lag (lower = more lag)

  document.addEventListener('mousemove', function(e) {
    targetX = e.clientX;
    targetY = e.clientY;
    // Direct update removed - handled by animation frame now
    // document.documentElement.style.setProperty('--cursor-x', e.clientX + 'px');
    // document.documentElement.style.setProperty('--cursor-y', e.clientY + 'px');
  });

  function updateOrbPosition() {
    // Calculate difference
    let dx = targetX - currentX;
    let dy = targetY - currentY;

    // Apply damping (inertia)
    currentX += dx * dampingFactor;
    currentY += dy * dampingFactor;

    // Update CSS variables
    orbElement.style.setProperty('--cursor-x', currentX + 'px');
    orbElement.style.setProperty('--cursor-y', currentY + 'px');

    // Continue the loop
    requestAnimationFrame(updateOrbPosition);
  }

  // Start the animation loop
  requestAnimationFrame(updateOrbPosition);

  /**
   * Text Selection Shape Change (Experimental)
   */
  let isMouseDown = false;
  let isSelecting = false;

  document.body.addEventListener('mousedown', () => {
    isMouseDown = true;
    isSelecting = false; // Reset selection flag on new mousedown
  });

  document.body.addEventListener('mousemove', () => {
    // If mouse moves while button is down, assume potential selection
    if (isMouseDown) {
      isSelecting = true; 
      // Check if already has class to avoid constant toggling
      if (!document.body.classList.contains('is-selecting-text')) {
         document.body.classList.add('is-selecting-text');
      }
    }
  });

  document.body.addEventListener('mouseup', () => {
    isMouseDown = false;
    // Only remove class if we were potentially selecting
    if (isSelecting) {
       document.body.classList.remove('is-selecting-text');
    }
    isSelecting = false; // Reset flag
  });

  // Also remove class if selection ends outside the window
  document.addEventListener('mouseleave', () => {
     if (isMouseDown || isSelecting) {
         document.body.classList.remove('is-selecting-text');
         isMouseDown = false;
         isSelecting = false;
     }
  });

})();