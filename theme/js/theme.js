// Dark-mode toggle (persisted in localStorage) and BibTeX expanders.

document.getElementById("theme-toggle").addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  localStorage.setItem("theme", next);
});

document.querySelectorAll(".bibtex-toggle").forEach((toggle) => {
  toggle.addEventListener("click", (event) => {
    event.preventDefault();
    const target = document.getElementById(toggle.dataset.target);
    target.hidden = !target.hidden;
  });
});
