# Security policy

## Supported versions

Only the latest version on the `main` branch, which is what the live demo at
<https://crop-disease-xai.onrender.com/> runs, receives fixes.

## Reporting a vulnerability

Please do **not** open a public issue for a security problem. Instead, use GitHub's private
vulnerability reporting: on the repository page, open **Security → Report a vulnerability**.

Include what you found, how to reproduce it, and the impact you expect. You will get a reply as soon
as the maintainers can look at it; this is a volunteer research project, so please allow some time.

## Scope

The web app is a static site: photos are processed in the visitor's browser and nothing is uploaded,
so there is no server-side data to protect. Relevant reports include, for example, script injection
through the app's pages, weaknesses in the Content-Security-Policy (`web/security-headers.js`), or a
compromised dependency.
