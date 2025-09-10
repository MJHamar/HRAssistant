/*
 * Minimal card factory with expand/collapse and lifecycle hooks.
 * No layout description required. Hooks receive the provided `id` and can
 * locate the card parts using: [data-card-id="<id>"] .card-{header|body|footer}
 *
 * Usage (later, when wiring):
 *   const el = HRCard.makeCard({ id: 'job-1', onInit, onExpand, onCollapse });
 *   list.appendChild(el);
 */
function toggleExpanded(root, expanded) {
    const header = root.querySelector('.card-header');
    const body = root.querySelector('.card-body');
    const footer = root.querySelector('.card-footer');
    if (expanded) {
        root.classList.add('expanded');
        // if (header) header.setAttribute('aria-expanded', 'true');
    } else {
        root.classList.remove('expanded');
        if (header) header.setAttribute('aria-expanded', 'false');
    }
}

function makeCard({ id, onInit, onExpand, onCollapse, initiallyExpanded = false } = {}) {
    if (!id) throw new Error('makeCard: `id` is required');

    // Root
    const root = document.createElement('div');
    root.className = 'card';
    // Use explicit attribute so callers can query with [data-card-id]
    root.setAttribute('data-card-id', String(id));

    // Header
    const header = document.createElement('div');
    header.className = 'card-header';
    header.setAttribute('role', 'button');
    header.setAttribute('aria-expanded', 'false');
    header.tabIndex = 0;

    // Body
    const body = document.createElement('div');
    body.className = 'card-body';

    // Footer
    const footer = document.createElement('div');
    footer.className = 'card-footer';

    // Compose
    root.appendChild(header);
    root.appendChild(body);
    root.appendChild(footer);

    // Internal state
    let didInit = false;

    // Wire toggle
    const doExpand = () => {
        toggleExpanded(root, true);
    if (typeof onExpand === 'function') onExpand(id, root);
    };
    const doCollapse = () => {
        toggleExpanded(root, false);
    if (typeof onCollapse === 'function') onCollapse(id, root);
    };

    const onToggle = () => {
        const isNowExpanded = !root.classList.contains('expanded');
        if (isNowExpanded) doExpand(); else doCollapse();
    };

    header.addEventListener('click', onToggle);
    header.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        onToggle();
        }
    });

    // Init once
    if (!didInit) {
        didInit = true;
    if (typeof onInit === 'function') onInit(id, root);
    }

    // Initial state
    if (initiallyExpanded) {
        doExpand();
    } else {
    toggleExpanded(root, false); // hide body & footer initially
    }

    return root;
}

// Expose globally (classic script usage)
window.HRCard = { makeCard };
window.makeCard = makeCard;