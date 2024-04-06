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

function findHeaders(state) {
    let allHeaders = Array.from(document.querySelectorAll(".main-content h1, .main-content h2"));

    for (let i = 0; i < allHeaders.length; i++) {
        let header = allHeaders[i];
        if (header.classList.contains("post-title")) {
            continue;
        }
        if (header.nodeName.toLowerCase() == "h1") {
            state.headers.primary.push(header);
            state.headers.secondary.push([]);
            continue;
        }

        if (state.headers.secondary.length == 0) {
            console.log("Found h2 before any h1. Skipping.");
            continue;
        }

        state.headers.secondary.at(-1).push(header);
    }
}

function buildTableOfContentsListElement(headerElem) {
    a_elem = document.createElement("a");
    a_elem.appendChild(document.createTextNode(headerElem.innerHTML));
    a_elem.setAttribute("href", "#" + headerElem.id);
    a_elem.addEventListener("click", toggleSidebar);

    li_elem = document.createElement("li");
    li_elem.append(a_elem);

    return li_elem;
}

function buildTableOfContents(state) {
    let contentsList = document.querySelector(".primary-contents-list");
    for (let primaryIdx = 0; primaryIdx < state.headers.primary.length; primaryIdx++) {
        let primaryListElem = buildTableOfContentsListElement(state.headers.primary[primaryIdx]);
        contentsList.append(primaryListElem);
        state.tocListElements.primary.push(primaryListElem);
        state.tocListElements.secondary.push([]);

        let secondaryHeaders = state.headers.secondary[primaryIdx];
        if (secondaryHeaders.length == 0) {
            continue;
        }

        let secondaryList = document.createElement("ul");
        secondaryList.classList.add("secondary-contents-list");
        primaryListElem.append(secondaryList);

        for (let secondaryIdx = 0; secondaryIdx < secondaryHeaders.length; secondaryIdx++) {
            let secondaryListElem = buildTableOfContentsListElement(secondaryHeaders[secondaryIdx]);
            secondaryList.append(secondaryListElem);
            state.tocListElements.secondary.at(-1).push(secondaryListElem);
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
    findHeaders(state);
    buildTableOfContents(state);
    console.log(state);
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