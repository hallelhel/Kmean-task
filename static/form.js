$(document).ready(function() {

	$('form').on('submit', function(event) {

		$.ajax({
			data : {
				number : $('#numberInput').val()
			},
			type : 'POST',
			url : '/update_cluster',

		})
		.done(function(data) {

			if (data.error) {
				$('#errorAlert').text(data.error).show();
				$('#successAlert').hide();

			}
			else {
				$('#img_div').html('<img src="data:image/png;base64,' + data.image + '" />');
                $('#errorAlert').hide();
			}

		});

		event.preventDefault();


	});



});
