Dropzone.autoDiscover = false;

function init() {
    let dz = new Dropzone("#dropzone", {
        url: "/",
        maxFiles: 1,
        addRemoveLinks: true,
        dictDefaultMessage: "Drop an image here or click to upload",
        autoProcessQueue: false
    });

    dz.on("addedfile", function() {
        if (dz.files[1] != null) {
            dz.removeFile(dz.files[0]);
        }
        $("#resultHolder").hide();
        $("#divClassTable").hide();
        $("#error").hide();
    });

    dz.on("complete", function(file) {
        let imageData = file.dataURL;
        let url = "http://127.0.0.1:5000/classify_image";

        $("#resultHolder").html("<div class='text-info'>Processing...</div>").show();

        $.post(url, {
            image_data: imageData
        }, function(data, status) {
            $("#resultHolder").empty();
            $("#divClassTable").hide();
            $("#error").hide();

            if (!data || data.length === 0) {
                $("#resultHolder").hide();
                $("#divClassTable").hide();
                $("#error").show();
                return;
            }

            let match = null;
            let bestScore = -1;
            for (let i = 0; i < data.length; ++i) {
                let maxScoreForThisClass = Math.max(...data[i].class_probability);
                if (maxScoreForThisClass > bestScore) {
                    match = data[i];
                    bestScore = maxScoreForThisClass;
                }
            }

            if (match) {
                $("#error").hide();
                $("#resultHolder").show();
                $("#divClassTable").show();
                // Show the actor card
                $(`[data-player]`).removeClass('border-primary');
                $(`[data-player="${match.class.toLowerCase()}"] .card`).addClass('border-primary');
                // Update probability scores
                let classDictionary = match.class_dictionary;
                for (let actorName in classDictionary) {
                    let index = classDictionary[actorName];
                    let probabilityScore = match.class_probability[index];
                    let elementName = "#score_" + actorName.toLowerCase();
                    $(elementName).html(probabilityScore !== undefined ? probabilityScore.toFixed(2) : "0.00");
                }
            }
        }).fail(function() {
            $("#resultHolder").hide();
            $("#divClassTable").hide();
            $("#error").show().html("Server error. Please try again.");
        });
    });

    $("#submitBtn").on('click', function(e) {
        if (dz.files.length === 0) {
            $("#error").show().html("Please upload an image before submitting.");
            return;
        }
        dz.processQueue();
    });
}

$(document).ready(function() {
    $("#error").hide();
    $("#resultHolder").hide();
    $("#divClassTable").hide();
    init();
});