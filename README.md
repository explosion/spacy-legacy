<a href="https://explosion.ai"><img src="https://explosion.ai/assets/img/logo.svg" width="125" height="125" align="right" /></a>

# spacy-legacy: Legacy functions and architectures for backwards compatibility

This package includes outdated registered functions for [spaCy](https://spacy.io) v3.x, for example model architectures, pipeline components and utilities. It's installed automatically as a dependency of spaCy, and allows us to provide backwards compaitbility, while keeping the core library tidy and up to date.

Whenever a new backwards-incompatible version of a registered function is available, e.g. `spacy.Tok2Vec.v1` &rarr; `spacy.Tok2Vec.v2`, the legacy version is moved to `spacy-legacy`, and exposed via [entry points](setup.cfg). This means that it will still be available if your config files use it, even though the core library only includes the latest version.
