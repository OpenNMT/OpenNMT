# Contributing

*Thanks for being there!*

There are many ways you can help in the OpenNMT project. This document will guide you through this process.

## Reporting issues

We use GitHub issues for bugs in the code that are **reproducible**. A good bug report should contain every information needed to reproduce it. Before opening a new issue, make sure to:

* **use the GitHub issue search** for existing and fixed bugs;
* **check if the issue has been fixed** in a more recent version;
* **isolate the problem** to give as much context as possible.

If you have questions on how to use the project or have trouble getting started with it, consider using [our forum](http://forum.opennmt.net/) instead.

## Making a pull request

*You want to share some code, that's great!*

### Before coding

If you are planning to submit a large pull request (e.g. a new feature, code refactoring), consider asking first on [the forum](http://forum.opennmt.net/) to confirm that this change is welcome.

### During the development

While we are open on the coding style, we would like the code to be consistent across the project. So please review and apply as much as possible our [style recommendations](STYLE.md).
If possible, write [automated tests](test/README.md)...

### Before submitting the pull request

When you are ready to submit your pull request, please review the following items:

* your branch must pass `luacheck` and the [automated tests](test/README.md)
* update the [CHANGELOG.md](CHANGELOG.md) file to list your feature or fix
* complete or update the documentation
* if you added new command line options, [update the options listing](docs/README.md#generating-options-listing) in the documentation
* if you are comfortable with Git, consider rebasing your branch to clean its history

### After submitting the pull request

Then, your changes will be reviewed and hopefully merged in a reasonable amount of time.

## Documenting

OpenNMT also has a lot of documentation material that can always be improved or completed. Visit the [online documentation](http://opennmt.net/OpenNMT/) and click on the edit button at the top of a page to submit your changes.

## Helping others

People often ask for help or suggestions on [our forum](http://forum.opennmt.net/). Consider visiting it regularly and help some of them!
