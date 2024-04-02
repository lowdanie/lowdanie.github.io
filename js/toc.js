// TODO
// * move navigation in header to right? what should be in the header?
// * what to do about white space on the right of the text on larger screens?
// * fix fonts / sizes / colors / line height
// * lines under header over footer. In small view, header line should disappear when opening sidepanel.
// * consistent link style
// * fix default.html vs post.html. also, fix mathjax <script> location
// * make css common constants --vars
// * contents should collapse in small view after clicking link
// * image sizes.
// * nicer blockquote css
// * fix index page

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

let currentActiveIndex = -1;

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
    let prev_timestamp = 0;
    return function () {
        current_timestamp = Date.now();
        if (current_timestamp > prev_timestamp + wait) {
            func();
            prev_timestamp = current_timestamp;
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