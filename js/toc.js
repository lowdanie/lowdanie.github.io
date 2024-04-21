let STATE = {
    headers: {
        primary: [],
        secondary: []
    },
    tocListElements: {
        primary: [],
        secondary: []
    },
    activeHeaderIndex: {
        primary: -1,
        secondary: -1
    }
};

let pageHeader = document.querySelector(".page-header");
let hamburger = document.querySelector(".hamburger");
let sidebar = document.querySelector(".sidebar");

let currentActiveIndex = -1;

function toggleSidebar() {
    if (window.getComputedStyle(hamburger).display == 'none') {
        return;
    }

    hamburger.classList.toggle("is-active");

    if (hamburger.classList.contains("is-active")) {
        sidebar.style.top = `${pageHeader.clientHeight}px`;
        sidebar.classList.remove("hidden");
        document.body.classList.add("noscroll");
    } else {
        sidebar.classList.add("hidden");
        document.body.classList.remove("noscroll");
    }
}

function loadTableOfContents(state) {
    let primaryContentsList = document.querySelector(".primary-contents-list");

    for (let primaryIdx = 0; primaryIdx < primaryContentsList.children.length; primaryIdx++) {
        let elem = primaryContentsList.children[primaryIdx];
        let linkElem = elem.querySelector('a');
        let headerId = linkElem.getAttribute('href');
        let headerElem = document.querySelector(headerId);

        linkElem.addEventListener("click", toggleSidebar);
        state.tocListElements.primary.push(elem);
        state.headers.primary.push(headerElem);

        state.tocListElements.secondary.push([]);
        state.headers.secondary.push([]);

        let secondaryContentsList = elem.querySelector('ul');
        if (secondaryContentsList == null) {
            continue;
        }

        for (let secondaryIdx = 0; secondaryIdx < secondaryContentsList.children.length; secondaryIdx++) {
            elem = secondaryContentsList.children[secondaryIdx];
            linkElem = elem.querySelector('a');
            headerId = linkElem.getAttribute('href');
            headerElem = document.querySelector(headerId);

            linkElem.addEventListener("click", toggleSidebar);
            state.tocListElements.secondary.at(-1).push(elem);
            state.headers.secondary.at(-1).push(headerElem);
        }
    }
}

function findActiveHeaderIndex(headers) {
    if (headers.length == 0) {
        return -1;
    }

    for (let i = 0; i < headers.length; i++) {
        if (headers[i].getBoundingClientRect().top > 200) {
            return i - 1;
        }
    }
    return headers.length;
}

function removeActiveHeaderClass(state) {
    let primaryIdx = state.activeHeaderIndex.primary;
    let secondaryIdx = state.activeHeaderIndex.secondary;

    if (primaryIdx == -1) {
        return;
    }

    let currentPrimary = state.tocListElements.primary[primaryIdx];
    currentPrimary.classList.remove("toc-active");
    currentPrimary.classList.remove("toc-primary-active");

    if (secondaryIdx == -1) {
        return;
    }

    let currentSecondary = state.tocListElements.secondary[primaryIdx][secondaryIdx];
    currentSecondary.classList.remove("toc-active");
}

function addActiveHeaderClass(state) {
    let primaryIdx = state.activeHeaderIndex.primary;
    let secondaryIdx = state.activeHeaderIndex.secondary;

    if (primaryIdx == -1) {
        return;
    }

    let currentPrimary = state.tocListElements.primary[primaryIdx];
    currentPrimary.scrollIntoView({ block: 'nearest', inline: 'nearest' });

    if (secondaryIdx == -1) {
        currentPrimary.classList.add("toc-active");
        return;
    }

    let currentSecondary = state.tocListElements.secondary[primaryIdx][secondaryIdx];
    currentPrimary.classList.add("toc-primary-active");
    currentSecondary.classList.add("toc-active");
}


function init(state) {
    loadTableOfContents(state);
}

function update(state) {
    let primaryActiveIndex = findActiveHeaderIndex(state.headers.primary);
    if (primaryActiveIndex == -1) {
        primaryActiveIndex = 0;
    } else if (primaryActiveIndex == state.headers.primary.length) {
        primaryActiveIndex = state.headers.primary.length - 1
    }

    let secondaryActiveIndex = findActiveHeaderIndex(state.headers.secondary[primaryActiveIndex]);
    if (secondaryActiveIndex == state.headers.secondary[primaryActiveIndex].length) {
        secondaryActiveIndex = state.headers.secondary[primaryActiveIndex].length - 1;
    }

    if ((primaryActiveIndex == state.activeHeaderIndex.primary) &&
        (secondaryActiveIndex == state.activeHeaderIndex.secondary)) {
        return;
    }

    removeActiveHeaderClass(state);

    state.activeHeaderIndex.primary = primaryActiveIndex;
    state.activeHeaderIndex.secondary = secondaryActiveIndex;

    addActiveHeaderClass(state);
}

function throttle(func, wait) {
    let waiting = false;

    return function () {
        if (!waiting) {
            waiting = true;
            setTimeout(() => {
                func();
                waiting = false;
            }, wait);
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    init(STATE);
    update(STATE);
});

window.addEventListener('scroll', throttle(() => {
    update(STATE);
}, 100));

hamburger.addEventListener("click", toggleSidebar);

const mql = window.matchMedia("(min-width: 1000px)");
mql.onchange = (e) => {
    if (e.matches) {
        sidebar.style.top = '0'
    }
}