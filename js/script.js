var hamburger = document.querySelector(".hamburger");
var sidebar = document.querySelector(".sidebar");

hamburger.addEventListener("click", function () {
    hamburger.classList.toggle("is-active");

    if (hamburger.classList.contains("is-active")) {
        sidebar.classList.remove("hidden");
        document.body.classList.add("noscroll");
    } else {
        sidebar.classList.add("hidden");
        document.body.classList.remove("noscroll");
    }
});
