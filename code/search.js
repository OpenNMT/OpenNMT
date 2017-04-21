function _ifNotFileProtocol(callback) {
    $(document).ready(function() {
        var url = window.location.href
        var protocol = url.split("/")[0];
        if (protocol != "file:") {
            callback();
        }
    });
}

function _searchForm() {
    return '<div class="searchForm"> \
        <form method="get" action="/search"> \
        <input id="query" name="query" type="search", placeholder="Input search terms here" autofocus /> \
        <input type="submit" value="Search"/> \
        </form> \
        </div>'
}

function addSearchFormHeader()
{
    _ifNotFileProtocol(function() {
        $("header > ul").append('<li>' + _searchForm() + '</li>');
    });
}

function addSearchFormBody(parentElement) {
    _ifNotFileProtocol(function() {
        $("body").prepend(_searchForm());
    });
}
