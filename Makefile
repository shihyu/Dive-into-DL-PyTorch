serve:
	@docsify serve docs

clean:
	rm -fr docs

github:
	@ghp-import docs -p -n
