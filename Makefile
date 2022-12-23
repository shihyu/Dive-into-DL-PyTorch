serve:
	@mdbook serve

build:
	@mdbook build

clean:
	rm -fr docs

github:
	@ghp-import docs -p -n
